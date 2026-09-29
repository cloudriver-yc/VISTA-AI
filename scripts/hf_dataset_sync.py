"""
Syncs VISTA-AI data and models with a private Hugging Face dataset repo, in the layout of src/paths.py.

  python scripts/hf_dataset_sync.py status [--remote]
  python scripts/hf_dataset_sync.py upload   [--parts ...] [--mirror] [--keep-legacy] [--prune-orphans] [--dry-run]
  python scripts/hf_dataset_sync.py download [--parts ...]

Parts (default: all):
  metadata         dialogues.jsonl (+ backups), split manifests, YouTube / uploads metadata
  features         data/features/*.pt (+ YouTube pre-label-swap backups): all you need to train (always mirrored)
  models           stage-1 weights + stage-2 adapters (always mirrored)
  synthetic-audio  TTS WAVs, per-turn TTS cache, augmented FLACs + manifest
  real-audio       YouTube audio + captions, uploaded call recordings

Not synced: data/tmp/ (scratch), data/noise/ (ESC-50, re-download with 03b_augment_audio.py --download-noise),
data/synthetic/augmented_preview/ (listening QA only).
"""
import os
import sys
import argparse
import subprocess
from fnmatch import fnmatch
from huggingface_hub import HfApi, snapshot_download, login, CommitOperationDelete
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

# Load environment variables from .env (such as HF_TOKEN)
load_dotenv()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_REPO_ID = "Vista-AI/CustomerServiceAudio"

# fnmatch patterns relative to the repo root ("*" also matches "/"), the same semantics huggingface_hub uses
PARTS = {
    "metadata": [
        os.path.join(paths.SYNTHETIC_DIR, "dialogues*.jsonl"),
        os.path.join(paths.DATA_DIR, "split_manifest*.json"),
        paths.YOUTUBE_METADATA_PATH,
        paths.UPLOADS_METADATA_PATH,
    ],
    "features": [
        f"{paths.FEATURES_DIR}/*",
        f"{paths.YOUTUBE_DIR}/features_before_label_swap/*",
    ],
    "models": [f"{paths.MODELS_DIR}/*"],
    "synthetic-audio": [
        f"{paths.TTS_DIR}/*",
        f"{paths.TTS_TURN_CACHE_DIR}/*",
        f"{paths.AUGMENTED_DIR}/*",
    ],
    "real-audio": [
        f"{paths.YOUTUBE_AUDIO_DIR}/*",
        f"{paths.YOUTUBE_CAPTIONS_DIR}/*",
        f"{paths.UPLOADS_AUDIO_DIR}/*",
    ],
}
IGNORE = ["*.DS_Store", "*/__pycache__/*", "*.tmp"]
# Pre-restructure layout (see scripts/migrate_data_layout.py). On upload, legacy files that already exist locally
# in the new layout are pruned from the Hub; legacy files with no local copy ("orphans") are only reported,
# unless --prune-orphans is given.
LEGACY = ["data/audio/*", "data/raw/*", "data/train_test_split*.json", "data/youtube_metadata.jsonl"]
# Parts whose Hub copy always mirrors the local one: every features/*.pt on disk ends up in training via
# --rebuild_split, and app.py loads any adapter it finds, so stale files must not come back on download
ALWAYS_MIRROR = {"features", "models"}


def matches(path, patterns):
    return any(fnmatch(path, p) for p in patterns)


def list_local(part):
    """Repo-relative POSIX paths of local files belonging to a part."""
    roots = {p.split("*")[0].rsplit("/", 1)[0] for p in PARTS[part]}
    files = set()
    for root in roots:
        abs_root = os.path.join(REPO_ROOT, root)
        if os.path.isfile(abs_root):
            files.add(root)
            continue
        for dirpath, _, filenames in os.walk(abs_root):
            for name in filenames:
                rel = os.path.relpath(os.path.join(dirpath, name), REPO_ROOT).replace(os.sep, "/")
                files.add(rel)
    return sorted(f for f in files if matches(f, PARTS[part]) and not matches(f, IGNORE))


def size_of(files):
    return sum(os.path.getsize(os.path.join(REPO_ROOT, f)) for f in files)


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def legacy_to_current(path):
    """Where a legacy-layout file lives in the current layout (mirrors scripts/migrate_data_layout.py)."""
    name = os.path.basename(path)
    if path.startswith("data/audio/out/"):
        return f"{paths.TTS_DIR}/{name}"
    if path.startswith("data/audio/tmp/") and "_turn_" in name:
        return f"{paths.TTS_TURN_CACHE_DIR}/{name}"
    if path.startswith("data/audio/youtube/captions/"):
        return f"{paths.YOUTUBE_CAPTIONS_DIR}/{name}"
    if path.startswith("data/audio/youtube/"):
        return f"{paths.YOUTUBE_AUDIO_DIR}/{name}"
    if path.startswith("data/audio/uploads/"):
        stem, ext = os.path.splitext(name)
        return f"{paths.UPLOADS_AUDIO_DIR}/{stem.replace('.', '_')}{ext}"
    if path.startswith("data/raw/"):
        return f"{paths.SYNTHETIC_DIR}/{name}"
    if path.startswith("data/train_test_split"):
        return f"{paths.DATA_DIR}/{name.replace('train_test_split', 'split_manifest')}"
    if path == "data/youtube_metadata.jsonl":
        return paths.YOUTUBE_METADATA_PATH
    return None


def split_legacy(remote_files):
    """Legacy Hub files, split into those with a local copy in the new layout and orphans without one."""
    migrated, orphans = [], []
    for f in remote_files:
        if matches(f, LEGACY):
            current = legacy_to_current(f)
            (migrated if current and os.path.exists(os.path.join(REPO_ROOT, current)) else orphans).append(f)
    return migrated, orphans


def list_remote(api, repo_id):
    try:
        return api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as e:
        print(f"⚠️ Could not list '{repo_id}' on the Hub ({type(e).__name__}: {e}).")
        return None


def authenticate():
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
        print("🔐 Authenticated with Hugging Face via HF_TOKEN.")
    else:
        print("⚠️ No HF_TOKEN in .env. For a private repo or any upload, add HF_TOKEN=... to .env "
              "or run 'huggingface-cli login'.")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
def print_status(repo_id=None, remote=False):
    print("\n" + "=" * 78)
    print("📦 LOCAL INVENTORY (layout of src/paths.py)")
    print("=" * 78)
    for part in PARTS:
        files = list_local(part)
        print(f"• {part:<16} {len(files):>6} files  {human(size_of(files)):>9}")
    models = [os.path.basename(f) for f in list_local("models")]
    adapters = [m for m in models if m.startswith("csat_adapter")]
    print(f"  models: {len(models) - len(adapters)} stage-1 weight files, stage-2 adapters: {', '.join(adapters) or 'none'}")
    for label, path in [("dialogues.jsonl", paths.DIALOGUES_PATH), ("split_manifest.json", paths.SPLIT_MANIFEST_PATH),
                        ("youtube metadata", paths.YOUTUBE_METADATA_PATH), ("uploads metadata", paths.UPLOADS_METADATA_PATH)]:
        print(f"  {'✅' if os.path.exists(os.path.join(REPO_ROOT, path)) else '❌'} {label}")
    print("  Not synced: data/tmp/, data/noise/ (re-download ESC-50), data/synthetic/augmented_preview/")

    if remote:
        api = HfApi()
        remote_files = list_remote(api, repo_id)
        if remote_files is not None:
            print("\n" + "=" * 78)
            print(f"☁️  HUB ({repo_id})")
            print("=" * 78)
            for part in PARTS:
                local = set(list_local(part))
                on_hub = {f for f in remote_files if matches(f, PARTS[part])}
                print(f"• {part:<16} {len(on_hub):>6} on Hub | {len(local - on_hub):>5} local-only | {len(on_hub - local):>5} Hub-only")
            migrated, orphans = split_legacy(remote_files)
            print(f"• legacy layout    {len(migrated) + len(orphans):>6} files still on the Hub: {len(migrated)} have a local copy "
                  f"in the new layout (pruned on upload), {len(orphans)} orphans (kept unless --prune-orphans)")
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------
def upload(repo_id, parts, mirror=False, keep_legacy=False, prune_orphans=False, dry_run=False, commit_message=None):
    api = HfApi()
    if not dry_run:
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)
        except Exception as e:
            print(f"⚠️ Could not create/verify '{repo_id}': {e}\nMake sure your HF_TOKEN has write access.")
    remote_files = list_remote(api, repo_id) or []
    commit_desc = commit_message or "Sync VISTA-AI data and models"

    # 1. Plan deletions: legacy layout, stale files in mirrored parts
    to_delete = set()
    migrated, orphans = split_legacy(remote_files)
    if not keep_legacy:
        # Only prune a legacy file when its new-layout copy is uploaded in this same run
        selected = [p for part in parts for p in PARTS[part]]
        to_delete |= {f for f in migrated if matches(legacy_to_current(f), selected)}
        if prune_orphans:
            to_delete |= set(orphans)
    for part in parts:
        if part in ALWAYS_MIRROR or mirror:
            local = set(list_local(part))
            if not local:
                print(f"⚠️ No local files for '{part}'; not deleting its Hub copy.")
                continue
            to_delete |= {f for f in remote_files if matches(f, PARTS[part]) and f not in local}

    # 2. Show the plan
    print(f"\n🚀 Upload plan for '{repo_id}'{' (dry run)' if dry_run else ''}:")
    for part in parts:
        files = list_local(part)
        print(f"  • {part:<16} {len(files):>6} files  {human(size_of(files)):>9}"
              + ("  (mirrored)" if part in ALWAYS_MIRROR or mirror else ""))
    legacy_count = sum(matches(f, LEGACY) for f in to_delete)
    print(f"  • delete on Hub: {len(to_delete)} files ({legacy_count} from the legacy layout, "
          f"{len(to_delete) - legacy_count} no longer present locally)")
    for f in sorted(to_delete)[:15]:
        print(f"      - {f}")
    if len(to_delete) > 15:
        print(f"      … and {len(to_delete) - 15} more")
    kept_orphans = [f for f in orphans if f not in to_delete]
    if kept_orphans:
        print(f"  • kept on Hub: {len(kept_orphans)} legacy files with no local copy (e.g. {kept_orphans[0]}). "
              "Add --prune-orphans to remove them.")
    if dry_run:
        print("\nDry run: nothing was uploaded or deleted.")
        return

    # 3. Upload each part as its own commit (unchanged files are skipped by content hash)
    for part in parts:
        if not list_local(part):
            print(f"⏭️  '{part}': no local files, skipped.")
            continue
        print(f"⏳ Uploading '{part}'...")
        # Scope the upload to the part's top-level folder (data/ or models/) so .venv etc. are never scanned
        top = PARTS[part][0].split("/", 1)[0]
        api.upload_folder(
            folder_path=os.path.join(REPO_ROOT, top),
            path_in_repo=top,
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[p.split("/", 1)[1] for p in PARTS[part]],
            ignore_patterns=IGNORE,
            commit_message=f"{commit_desc} ({part})",
        )
        print(f"✅ '{part}' uploaded.")

    # 4. One commit for all deletions
    if to_delete:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=[CommitOperationDelete(path_in_repo=f) for f in sorted(to_delete)],
            commit_message=f"{commit_desc} (remove {len(to_delete)} stale/legacy files)",
        )
        print(f"🗑️  Removed {len(to_delete)} stale/legacy files from the Hub (still recoverable from the repo history).")
    print("\n🎉 Hub is in sync with the local layout.")


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------
def download(repo_id, parts):
    patterns = [p for part in parts for p in PARTS[part]]
    print(f"📥 Downloading {', '.join(parts)} from '{repo_id}'...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=REPO_ROOT,
        # Legacy patterns too, so a Hub copy that still has the old layout is fetched and then migrated
        allow_patterns=patterns + LEGACY,
        ignore_patterns=IGNORE + ["data/tmp/*", "data/audio/tmp/*"],
    )
    print("✅ Download complete.")
    # Normalize a legacy layout into src/paths.py's layout (no-op if nothing legacy was downloaded)
    subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "migrate_data_layout.py")], check=True, cwd=REPO_ROOT)

    if "models" in parts:
        remote_models = {f for f in (list_remote(HfApi(), repo_id) or []) if matches(f, PARTS["models"])}
        stale = [f for f in list_local("models") if remote_models and f not in remote_models]
        if stale:
            print("⚠️ These local model files are not on the Hub (e.g. an adapter the stage-2 guard rejected). "
                  "app.py still loads them; delete them if they are stale:")
            for f in stale:
                print(f"   {f}")
    print_status()


def main():
    parser = argparse.ArgumentParser(description="Hugging Face sync for VISTA-AI data and models (src/paths.py layout)")
    parser.add_argument("action", choices=["upload", "download", "status"])
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo ID")
    parser.add_argument("--parts", nargs="+", choices=list(PARTS), default=list(PARTS),
                        help="What to sync (default: all). E.g. '--parts metadata features models' is enough to train and run the app")
    parser.add_argument("--mirror", action="store_true",
                        help="upload: also delete Hub files of the selected parts that no longer exist locally (features and models are always mirrored)")
    parser.add_argument("--keep-legacy", action="store_true", help="upload: don't prune the pre-restructure layout from the Hub")
    parser.add_argument("--prune-orphans", action="store_true",
                        help="upload: also delete legacy-layout Hub files that have no local copy in the new layout")
    parser.add_argument("--dry-run", action="store_true", help="upload: show what would be uploaded/deleted without changing the Hub")
    parser.add_argument("--remote", action="store_true", help="status: also compare against the Hub")
    parser.add_argument("--commit-message", default=None, help="upload: custom commit message prefix")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)

    if args.action == "status":
        if args.remote:
            authenticate()
        print_status(args.repo_id, remote=args.remote)
        return

    authenticate()
    if args.action == "upload":
        upload(args.repo_id, args.parts, mirror=args.mirror, keep_legacy=args.keep_legacy,
               prune_orphans=args.prune_orphans, dry_run=args.dry_run, commit_message=args.commit_message)
    else:
        download(args.repo_id, args.parts)


if __name__ == "__main__":
    main()
