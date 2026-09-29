"""
Migrates the legacy data/ layout to the provenance-based layout defined in src/paths.py.

Legacy                                  ->  New
data/raw/dialogues*.jsonl               ->  data/synthetic/dialogues*.jsonl
data/audio/out/*.wav                    ->  data/synthetic/tts/
data/audio/tmp/*_turn_*.npy             ->  data/synthetic/tts_turn_cache/
data/audio/tmp/* (app scratch)          ->  data/tmp/
data/audio/augmented[_preview]/         ->  data/synthetic/augmented[_preview]/
data/audio/youtube/*.wav                ->  data/real/youtube/audio/
data/audio/youtube/captions/            ->  data/real/youtube/captions/
data/youtube_metadata.jsonl             ->  data/real/youtube/metadata.jsonl
data/audio/uploads/<ts>.<ext>           ->  data/real/uploads/audio/<ts with '.'->'_'>.<ext>  (+ metadata.jsonl)
data/train_test_split[.v1].json         ->  data/split_manifest[.v1].json

Idempotent: safe to re-run (e.g. after `hf_dataset_sync.py download` pulls the legacy layout).
Features (data/features/) and the paths stored in the split manifest are unchanged.
"""
import os
import sys
import glob
import json
import shutil
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

CLASS_NAMES = ["very_unsatisfied", "unsatisfied", "satisfied", "very_satisfied"]


class Migrator:
    def __init__(self, dry_run):
        self.dry_run = dry_run
        self.moved = 0
        self.conflicts = []

    def move(self, src, dst):
        if not os.path.exists(src):
            return
        if os.path.exists(dst):
            if os.path.isfile(src) and os.path.getsize(src) == os.path.getsize(dst):
                # Already migrated (same file present at destination); drop the legacy copy
                if not self.dry_run:
                    os.remove(src)
                return
            self.conflicts.append((src, dst))
            return
        print(f"  {src}  ->  {dst}")
        self.moved += 1
        if not self.dry_run:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)

    def move_contents(self, src_dir, dst_dir, pattern="*", rename=None):
        for src in sorted(glob.glob(os.path.join(src_dir, pattern))):
            name = os.path.basename(src)
            if name == ".DS_Store" or os.path.isdir(src):
                continue
            self.move(src, os.path.join(dst_dir, rename(name) if rename else name))


def upload_stem(filename):
    """'1735404531.458927.mp3' -> '1735404531_458927' (matches features/upload_<stem>.pt)."""
    stem, _ = os.path.splitext(filename)
    return stem.replace(".", "_")


def write_uploads_metadata(dry_run):
    """Uploaded calls had no label file (labels lived only in features/upload_*.pt); write one."""
    import torch

    records = []
    for audio in sorted(glob.glob(os.path.join(paths.UPLOADS_AUDIO_DIR, "*"))):
        stem = os.path.splitext(os.path.basename(audio))[0]
        feature = os.path.join(paths.FEATURES_DIR, f"upload_{stem}.pt")
        rec = {"upload_id": stem, "audio_file": os.path.basename(audio), "feature_file": os.path.basename(feature)}
        if os.path.exists(feature):
            label = torch.load(feature, map_location="cpu")["label"]
            label = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
            rec.update({"label": label, "label_name": CLASS_NAMES[label]})
        records.append(rec)
    content = "".join(json.dumps(rec) + "\n" for rec in records)
    if not records:
        return
    if os.path.exists(paths.UPLOADS_METADATA_PATH):
        with open(paths.UPLOADS_METADATA_PATH, encoding="utf-8") as f:
            if f.read() == content:
                return
    print(f"  write {paths.UPLOADS_METADATA_PATH} ({len(records)} uploads)")
    if not dry_run:
        os.makedirs(os.path.dirname(paths.UPLOADS_METADATA_PATH), exist_ok=True)
        with open(paths.UPLOADS_METADATA_PATH, "w", encoding="utf-8") as f:
            f.write(content)


def remove_empty_dirs(root, dry_run):
    for dirpath, _, _ in sorted(os.walk(root), key=lambda x: -len(x[0])):
        entries = [e for e in os.listdir(dirpath) if e != ".DS_Store"]
        if not entries:
            print(f"  rmdir {dirpath}")
            if not dry_run:
                shutil.rmtree(dirpath)


def main():
    parser = argparse.ArgumentParser(description="Migrate data/ to the provenance-based layout")
    parser.add_argument("--dry-run", action="store_true", help="Print the moves without touching any files")
    args = parser.parse_args()
    m = Migrator(args.dry_run)

    print("🗂️  Migrating data/ layout" + (" (dry run)" if args.dry_run else "") + "...")
    # Synthetic pool
    m.move_contents("data/raw", paths.SYNTHETIC_DIR, "dialogues*.jsonl")
    m.move_contents("data/audio/out", paths.TTS_DIR)
    m.move_contents("data/audio/tmp", paths.TTS_TURN_CACHE_DIR, "*_turn_*.npy")
    m.move_contents("data/audio/tmp", paths.TMP_DIR)
    m.move_contents("data/audio/augmented", paths.AUGMENTED_DIR)
    m.move_contents("data/audio/augmented_preview", paths.AUGMENTED_PREVIEW_DIR)
    # Real-world pool
    m.move_contents("data/audio/youtube", paths.YOUTUBE_AUDIO_DIR, "*.wav")
    m.move_contents("data/audio/youtube/captions", paths.YOUTUBE_CAPTIONS_DIR)
    m.move("data/youtube_metadata.jsonl", paths.YOUTUBE_METADATA_PATH)
    m.move_contents("data/audio/uploads", paths.UPLOADS_AUDIO_DIR,
                    rename=lambda n: upload_stem(n) + os.path.splitext(n)[1])
    # Split manifest
    m.move("data/train_test_split.json", paths.SPLIT_MANIFEST_PATH)
    m.move("data/train_test_split.v1.json", paths.SPLIT_MANIFEST_PATH.replace(".json", ".v1.json"))

    write_uploads_metadata(args.dry_run)
    for legacy in ["data/audio", "data/raw"]:
        if os.path.isdir(legacy):
            remove_empty_dirs(legacy, args.dry_run)

    print(f"\n✅ {m.moved} files {'would be ' if args.dry_run else ''}moved.")
    if m.conflicts:
        print(f"⚠️ {len(m.conflicts)} conflicts left in place (destination exists with different content):")
        for src, dst in m.conflicts:
            print(f"  {src}  vs  {dst}")


if __name__ == "__main__":
    main()
