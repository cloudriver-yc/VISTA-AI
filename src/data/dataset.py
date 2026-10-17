import os
import glob
import json
import random
import shutil
import hashlib
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import paths
from paths import SPLIT_MANIFEST_PATH, DIALOGUES_PATH, FEATURES_DIR, FINETUNE_PREFIX, TEST_PREFIX
SPLITS = paths.FEATURE_SPLITS  # ("train", "val", "finetune", "test"); each is a folder under data/features/


def _source_id(basename):
    """'dial_001_x__aug03.pt' -> 'dial_001_x'; augmented variants share their source's id."""
    return os.path.splitext(basename)[0].split("__aug")[0]


def _script_groups():
    """Maps dialogue_id -> hash of its turn texts, so re-voiced copies of one script share a group."""
    groups = {}
    if os.path.exists(DIALOGUES_PATH):
        with open(DIALOGUES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                joined = "||".join(t["text"].strip().lower() for t in d["turns"])
                groups[d["dialogue_id"]] = hashlib.sha1(joined.encode("utf-8")).hexdigest()
    return groups


def _read_folders():
    return {split: sorted(glob.glob(os.path.join(paths.feature_split_dir(split), "*.pt"))) for split in SPLITS}


def _write_manifest(splits):
    """split_manifest.json is a record of the split folders, kept for tooling and the Hub; the folders are the truth."""
    if os.path.exists(SPLIT_MANIFEST_PATH):
        try:
            with open(SPLIT_MANIFEST_PATH, "r", encoding="utf-8") as f:
                if json.load(f) == splits:
                    return
        except Exception:
            pass
    os.makedirs(os.path.dirname(SPLIT_MANIFEST_PATH), exist_ok=True)
    with open(SPLIT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


def organize_flat_features(dry_run=False):
    """
    Moves feature files left directly in data/features/ (the pre-split-folder layout, or an old Hub copy)
    into their split folder. The split comes from split_manifest.json when it lists the file under one of the
    four splits; otherwise uploads go to finetune/, YouTube calls to test/ and synthetic files to unassigned/.
    A flat file whose name already exists in a folder is a stale duplicate: it is dropped if the sizes match
    and left in place (and reported) if not. Returns the number of files moved.
    """
    flat = sorted(glob.glob(os.path.join(FEATURES_DIR, "*.pt")))
    if not flat:
        return 0
    recorded = {}
    if os.path.exists(SPLIT_MANIFEST_PATH):
        try:
            with open(SPLIT_MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if all(k in manifest for k in SPLITS):  # older {"train","test"} manifests mixed sources; ignore them
                recorded = {os.path.basename(p): split for split in SPLITS for p in manifest[split]}
        except Exception:
            pass
    moved, conflicts = 0, []
    for src in flat:
        name = os.path.basename(src)
        existing = paths.find_feature(name)
        if existing:
            if os.path.getsize(existing) == os.path.getsize(src):
                if not dry_run:
                    os.remove(src)
            else:
                conflicts.append((src, existing))
            continue
        dst_dir = paths.feature_split_dir(recorded[name]) if name in recorded else paths.feature_home(name)
        moved += 1
        if not dry_run:
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src, os.path.join(dst_dir, name))
    if moved:
        print(f"🗂️ {'Would move' if dry_run else 'Moved'} {moved} feature files from {FEATURES_DIR}/ into their split folders.")
    for src, existing in conflicts:
        print(f"⚠️ {src} differs from {existing}; left in place (delete whichever is stale).")
    return moved


def _label_of(f):
    label = torch.load(f, map_location="cpu")["label"]
    return label.item() if isinstance(label, torch.Tensor) else int(label)


def _split_real_calls(originals, test_ratio, seed):
    """
    Stratified split of real calls (uploads + YouTube, originals only) into finetune / test.
    Per class, round(n * test_ratio) calls go to test, keeping at least one call on each side when a class
    has two or more. Returns {file: "finetune" | "test"}.
    """
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for f in originals:
        by_label[_label_of(f)].append(f)
    target = {}
    for label in sorted(by_label):
        files = sorted(by_label[label], key=os.path.basename)
        rng.shuffle(files)
        n = len(files)
        n_test = min(max(round(n * test_ratio), 1), n - 1) if n > 1 else n
        for i, f in enumerate(files):
            target[f] = "test" if i < n_test else "finetune"
    return target


def get_or_create_splits(features_dir=FEATURES_DIR, val_ratio=0.15, seed=42, rebuild=False,
                         resplit_real=False, real_test_ratio=0.5):
    """
    Returns {"train","val","finetune","test"} -> feature files, read from data/features/<split>/.
    - train/val: synthetic TTS dialogues (dial*) and their augmented variants (stage 1: pretraining).
    - finetune: real calls (upload_* / yt_*) that train the stage-2 adapter, plus their augmented copies.
    - test: real calls for the final evaluation (stage 3), never trained on. Originals only.
    Newly extracted files wait in data/features/unassigned/ and are not used until rebuild=True
    (src/train.py --rebuild_split), which sorts every file into its folder:
    - synthetic: val_ratio of the unique *scripts* per class are held out as val, and every re-voiced copy
      (dial_aug_*) and augmented variant (__augNN) of a script stays on the same side, so no dialogue text is
      shared between train and val;
    - real originals keep their folder (so a call moved by hand stays put); new ones go to their default
      (uploads -> finetune, YouTube -> test). With resplit_real=True they are instead re-split, stratified by
      class, with real_test_ratio of each class in test (_split_real_calls);
    - augmented copies of a real call follow their original into finetune/, or into excluded/ when the original
      is a test call, so no version of a test call ever reaches stage 2.
    """
    if os.path.abspath(features_dir) != os.path.abspath(FEATURES_DIR):
        raise ValueError(f"Feature files live in {FEATURES_DIR}/<split>/ (see src/paths.py); got {features_dir}")
    organize_flat_features()
    unassigned = sorted(glob.glob(os.path.join(paths.FEATURES_UNASSIGNED_DIR, "*.pt")))
    splits = _read_folders()
    if not rebuild and unassigned and not (splits["train"] or splits["val"]):
        print("ℹ️ No synthetic train/val split yet; creating one.")
        rebuild = True
    if not (rebuild or resplit_real):
        if unassigned:
            print(f"ℹ️ {len(unassigned)} feature files in {paths.FEATURES_UNASSIGNED_DIR}/ are not used yet; "
                  "run src/train.py --rebuild_split to sort them into their split.")
        _write_manifest(splits)
        return splits

    all_files = sorted(f for d in paths.feature_dirs() for f in glob.glob(os.path.join(d, "*.pt")))
    random.seed(seed)
    script_groups = _script_groups()

    target = {}  # file -> split folder name ("train", ..., "test", "excluded")
    # label -> group key -> files
    syn_groups = defaultdict(lambda: defaultdict(list))
    real_originals, real_copies = [], []

    for f in all_files:
        basename = os.path.basename(f)
        if basename.startswith((FINETUNE_PREFIX, TEST_PREFIX)):
            (real_copies if paths.is_augmented(basename) else real_originals).append(f)
            continue
        data = torch.load(f, map_location="cpu")
        label = data["label"].item() if isinstance(data["label"], torch.Tensor) else int(data["label"])
        source_id = data.get("source_id", _source_id(basename))
        group = script_groups.get(source_id, source_id)
        syn_groups[label][group].append(f)

    # Hold out val_ratio of the unique scripts per class (at least 1)
    for label in sorted(syn_groups):
        groups = sorted(syn_groups[label])
        random.shuffle(groups)
        n_val = max(1, round(len(groups) * val_ratio)) if len(groups) > 1 else 0
        for i, group in enumerate(groups):
            for f in syn_groups[label][group]:
                target[f] = "val" if i < n_val else "train"

    # Real calls: re-split, or keep their folder (new ones go to their default)
    if resplit_real:
        target.update(_split_real_calls(real_originals, real_test_ratio, seed))
    else:
        for f in real_originals:
            current = os.path.basename(os.path.dirname(f))
            target[f] = current if current in ("finetune", "test") else os.path.basename(paths.feature_home(os.path.basename(f)))
    original_split = {paths.feature_source_id(os.path.basename(f)): split
                      for f, split in target.items() if f in real_originals}
    for f in real_copies:
        split = original_split.get(paths.feature_source_id(os.path.basename(f)))
        target[f] = {"finetune": "finetune", "test": "excluded"}.get(split, "unassigned")

    moved = 0
    for f, split in target.items():
        dst = os.path.join(FEATURES_DIR, split, os.path.basename(f))
        if os.path.abspath(f) != os.path.abspath(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(f, dst)
            moved += 1
    if moved:
        print(f"🔀 Moved {moved} feature files between split folders.")

    splits = _read_folders()
    _check_no_leak(splits)
    _write_manifest(splits)
    return splits


def _check_no_leak(splits):
    """Fails loudly if a test call (or any copy of it) is also in finetune, or a copy sits in test."""
    finetune_ids = {paths.feature_source_id(os.path.basename(f)) for f in splits["finetune"]}
    test_ids = {paths.feature_source_id(os.path.basename(f)) for f in splits["test"]}
    shared = finetune_ids & test_ids
    copies_in_test = [f for f in splits["test"] if paths.is_augmented(os.path.basename(f))]
    if shared or copies_in_test:
        raise RuntimeError(f"Finetune/test leak: shared calls {sorted(shared)}, augmented copies in test {copies_in_test}")


class RealCSATDataset(Dataset):
    """
    Loads extracted multimodal PyTorch tensors with strict conversation-level isolation.
    """
    def __init__(self, features_dir=FEATURES_DIR, split="all", val_ratio=0.15, seed=42):
        self.split = split
        
        if split in SPLITS:
            splits = get_or_create_splits(features_dir=features_dir, val_ratio=val_ratio, seed=seed)
            self.files = splits[split]
        elif split == "all":
            dirs = [paths.feature_split_dir(x) for x in SPLITS] + [paths.FEATURES_UNASSIGNED_DIR]  # never excluded/
            self.files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "*.pt")))
        else:
            raise ValueError(f"Unknown split: '{split}'. Must be one of {SPLITS} or 'all'.")
            
        if len(self.files) == 0:
            raise RuntimeError(f"No feature files found for split='{split}' in {features_dir}!")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")
        
        # Dual Transformer format
        audio_embeds = data.get("audio_embeds", None)
        text_embeds = data.get("text_embeds", None)
        
        # Legacy Hybrid CNN format fallback
        if audio_embeds is None and "mel_spec" in data:
            audio_embeds = data.get("mel_spec")
        if text_embeds is None and "chunked_text" in data:
            text_embeds = data.get("chunked_text")
            
        label = data["label"]
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        return text_embeds, audio_embeds, label


def pad_collate(batch):
    """
    Custom collate_fn to pad varying sequence lengths across dialogues and generate padding masks.
    """
    text_embeds_list = [item[0] for item in batch]
    audio_embeds_list = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    batch_size = len(batch)
    lengths = [t.size(0) for t in text_embeds_list]
    max_len = max(lengths)
    
    # Pad sequences
    text_padded = pad_sequence(text_embeds_list, batch_first=True, padding_value=0.0)
    audio_padded = pad_sequence(audio_embeds_list, batch_first=True, padding_value=0.0)
    
    # Construct boolean padding mask (True where index >= sequence length)
    padding_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, l in enumerate(lengths):
        if l < max_len:
            padding_mask[i, l:] = True
            
    labels_stacked = torch.stack(labels)
    
    return text_padded, audio_padded, padding_mask, labels_stacked


