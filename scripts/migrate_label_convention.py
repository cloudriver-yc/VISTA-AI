"""
Migrates data and weights from the old "project" label convention to the everyday one.

Old (project)                                   New (everyday)
0 "Very Unsatisfied" = delighted, resolved   -> 0 "Very Unsatisfied" = angry, unresolved
3 "Very Satisfied"   = angry, unresolved     -> 3 "Very Satisfied"   = delighted, resolved
(1 "Unsatisfied" and 2 "Satisfied" are unchanged.)

The content of each call does not change, only which index/name its class has, so this swaps
indices 0 <-> 3 and the strings very_unsatisfied <-> very_satisfied:
  * feature files (data/features/*/*.pt): label 0 <-> 3, marked label_convention="everyday"
  * weights (models/*.pt): rows 0 and 3 of the 4-class output layer, so every model predicts exactly
    the same call as before under the new index (no retraining)
  * label strings: action_label in dialogues*.jsonl, label/label_name in the uploads metadata and
    the augmented-call manifests
Dialogue ids and file names keep their historical suffixes (e.g. dial_001_very_satisfied is an angry call).

Idempotent: feature files and manifest rows carry a label_convention marker, and models/label_convention.json
records the hash of each swapped weight file, so re-running (e.g. after `hf_dataset_sync.py download`
pulls old-convention copies) only touches what is still in the old convention. The git-tracked jsonl
files are migrated once and committed; pass --strings to swap them again (old copies only).
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

CONVENTION = "everyday"
SWAP_IDX = {0: 3, 3: 0}
SWAP_STR = {"very_unsatisfied": "very_satisfied", "very_satisfied": "very_unsatisfied"}
ACTION_LABEL_RE = re.compile(r'"action_label": "(very_unsatisfied|very_satisfied)"')
WEIGHT_MARKER = os.path.join(paths.MODELS_DIR, "label_convention.json")
OUTPUT_LAYERS = ("classifier.4", "out")  # V1/V2 head, ResidualAdapterHead


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def migrate_features(dry_run):
    changed = 0
    for f in sorted(glob.glob(os.path.join(paths.FEATURES_DIR, "*", "*.pt"))):
        data = torch.load(f, map_location="cpu")
        if data.get("label_convention") == CONVENTION:
            continue
        label = int(data["label"])
        new = SWAP_IDX.get(label, label)
        data["label"] = torch.tensor(new) if isinstance(data["label"], torch.Tensor) else new
        data["label_convention"] = CONVENTION
        changed += 1
        if not dry_run:
            torch.save(data, f)
    print(f"feature files: {changed} migrated")


def migrate_weights(dry_run):
    marker = json.load(open(WEIGHT_MARKER)) if os.path.exists(WEIGHT_MARKER) else {"convention": CONVENTION, "sha256": {}}
    changed = 0
    for f in sorted(glob.glob(os.path.join(paths.MODELS_DIR, "*.pt"))):
        name = os.path.basename(f)
        if marker["sha256"].get(name) == sha256(f):
            continue
        sd = torch.load(f, map_location="cpu")
        keys = [f"{layer}.{p}" for layer in OUTPUT_LAYERS for p in ("weight", "bias") if f"{layer}.{p}" in sd]
        if not keys or any(sd[k].shape[0] != 4 for k in keys):
            print(f"  skipped {name}: no 4-class output layer")
            continue
        for k in keys:
            sd[k] = sd[k][[3, 1, 2, 0]].clone()
        changed += 1
        if not dry_run:
            torch.save(sd, f)
            marker["sha256"][name] = sha256(f)
    if not dry_run:
        json.dump(marker, open(WEIGHT_MARKER, "w"), indent=2)
    print(f"weight files: {changed} migrated")


def rewrite_jsonl(path, fn, dry_run):
    if not os.path.exists(path):
        return 0
    rows = [json.loads(line) for line in open(path) if line.strip()]
    changed = sum(fn(r) for r in rows)
    if changed and not dry_run:
        with open(path, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"{path}: {changed} rows migrated")
    return changed


def swap_action_label_text(path, dry_run):
    """dialogues*.jsonl are git-tracked: swap the action_label values in place, leaving every other byte as is."""
    text = open(path, encoding="utf-8").read()
    new, changed = ACTION_LABEL_RE.subn(lambda m: f'"action_label": "{SWAP_STR[m.group(1)]}"', text)
    if changed and not dry_run:
        open(path, "w", encoding="utf-8").write(new)
    print(f"{path}: {changed} rows migrated")


def swap_metadata_label(row):
    """uploads/metadata.jsonl rows hold an int label and its name."""
    if "label" not in row:
        return 0
    row["label"] = SWAP_IDX.get(int(row["label"]), int(row["label"]))
    if row.get("label_name") in SWAP_STR:
        row["label_name"] = SWAP_STR[row["label_name"]]
    return 1


def swap_manifest_label(row):
    """Augmented-call manifests (untracked, synced with the audio): marked per row."""
    if row.get("label_convention") == CONVENTION:
        return 0
    if row.get("label") in SWAP_STR:
        row["label"] = SWAP_STR[row["label"]]
    row["label_convention"] = CONVENTION
    return 1


def main():
    parser = argparse.ArgumentParser(description="Swap the extreme CSAT classes to the everyday label convention")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strings", action="store_true",
                        help="Also swap the git-tracked label strings (dialogues*.jsonl, uploads/metadata.jsonl); "
                             "only for copies still in the old convention")
    args = parser.parse_args()

    migrate_features(args.dry_run)
    migrate_weights(args.dry_run)
    for path in (paths.AUGMENTED_MANIFEST_PATH, paths.UPLOADS_AUGMENTED_MANIFEST_PATH, paths.YOUTUBE_AUGMENTED_MANIFEST_PATH):
        rewrite_jsonl(path, swap_manifest_label, args.dry_run)
    if args.strings:
        for path in sorted(glob.glob(os.path.join(paths.SYNTHETIC_DIR, "dialogues*.jsonl"))):
            swap_action_label_text(path, args.dry_run)
        rewrite_jsonl(paths.UPLOADS_METADATA_PATH, swap_metadata_label, args.dry_run)


if __name__ == "__main__":
    main()
