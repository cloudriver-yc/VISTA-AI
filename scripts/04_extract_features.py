import os
import sys
import json
import argparse
import torch
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths
from feature_extraction import LABEL_MAP, extract_features_from_audio, get_device, load_extractors

AUDIO_DIR = paths.TTS_DIR
AUGMENTED_AUDIO_DIR = paths.AUGMENTED_DIR
AUGMENTED_MANIFEST_PATH = paths.AUGMENTED_MANIFEST_PATH
YOUTUBE_AUDIO_DIR = paths.YOUTUBE_AUDIO_DIR
JSON_PATH = paths.DIALOGUES_PATH
YOUTUBE_METADATA_PATH = paths.YOUTUBE_METADATA_PATH
OUTPUT_DIR = paths.FEATURES_DIR

def is_up_to_date(out_path, channel):
    """True if the feature file exists and was extracted with the same channel mode."""
    if not os.path.exists(out_path):
        return False
    return torch.load(out_path, map_location="cpu").get("channel") == channel


def main():
    parser = argparse.ArgumentParser(description="Whisper ASR -> WavLM + MPNet per-segment feature extraction")
    parser.add_argument("--only", default="syn,aug,yt",
                        help="Comma-separated sources to process: syn (data/synthetic/tts), aug (data/synthetic/augmented), yt (data/real/youtube/audio)")
    parser.add_argument("--channel", choices=["mix", "customer"], default="mix",
                        help="How stereo synthetic audio is reduced to mono (augmented and YouTube audio is already mono)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files whose features already exist with the same --channel")
    args = parser.parse_args()
    only = {x.strip() for x in args.only.split(",") if x.strip()}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = get_device()
    print(f"🚀 Feature extraction device: {device}")

    # 1. Initialize Feature Extractors
    extractors = load_extractors(device)

    failed = []

    def process(jobs, desc):
        """jobs: list of (audio_path, out_name, label, extra_fields). Returns (processed, skipped)."""
        processed, skipped = 0, 0
        for audio_path, out_name, label, extra in tqdm(jobs, desc=desc):
            out_path = os.path.join(OUTPUT_DIR, out_name)
            channel_tag = extra.get("channel", args.channel)
            if args.skip_existing and is_up_to_date(out_path, channel_tag):
                skipped += 1
                continue
            try:
                audio_emb, text_emb = extract_features_from_audio(audio_path, channel=args.channel, **extractors)
            except Exception as e:  # one bad file shouldn't end a multi-hour run; re-run with --skip-existing
                failed.append(audio_path)
                tqdm.write(f"❌ {os.path.basename(audio_path)}: {type(e).__name__}: {e}")
                continue
            torch.save({
                "audio_embeds": audio_emb,
                "text_embeds": text_emb,
                "label": torch.tensor(label, dtype=torch.long),
                "channel": channel_tag,
                **extra
            }, out_path)
            processed += 1
        return processed, skipped

    summary = {}

    # 2. Process Synthesized Audio (data/synthetic/tts/*.wav)
    if "syn" in only:
        dataset = {}
        if os.path.exists(JSON_PATH):
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    dataset[data["dialogue_id"]] = LABEL_MAP[data["action_label"]]
        syn_wav_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]) if os.path.exists(AUDIO_DIR) else []
        jobs = [(os.path.join(AUDIO_DIR, f), f"{f[:-4]}.pt", dataset[f[:-4]], {"source_id": f[:-4]})
                for f in syn_wav_files if f[:-4] in dataset]
        print(f"\n📂 Processing {len(jobs)} Synthesized Audio Files from '{AUDIO_DIR}' (channel={args.channel})...")
        summary["Synthesized"] = process(jobs, "Synthesized Audio")

    # 3. Process Augmented Synthetic Audio (data/synthetic/augmented/*.flac)
    if "aug" in only:
        jobs = []
        if os.path.exists(AUGMENTED_MANIFEST_PATH):
            with open(AUGMENTED_MANIFEST_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    rec = json.loads(line)
                    audio_path = os.path.join(AUGMENTED_AUDIO_DIR, rec["file"])
                    if os.path.exists(audio_path):
                        out_name = os.path.splitext(rec["file"])[0] + ".pt"
                        # Augmented clips are already mono; their channel mode is the one used at augmentation time
                        jobs.append((audio_path, out_name, LABEL_MAP[rec["label"]],
                                     {"source_id": rec["source_id"], "channel": rec["params"]["channel"]}))
        print(f"\n🎛️ Processing {len(jobs)} Augmented Audio Files from '{AUGMENTED_AUDIO_DIR}'...")
        summary["Augmented"] = process(jobs, "Augmented Audio")

    # 4. Process Real-World YouTube Audio (data/real/youtube/audio/*.wav)
    if "yt" in only:
        yt_metadata = {}
        if os.path.exists(YOUTUBE_METADATA_PATH):
            with open(YOUTUBE_METADATA_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    rec = json.loads(line)
                    yt_metadata[rec["video_id"]] = rec["label"]
        yt_wav_files = sorted([f for f in os.listdir(YOUTUBE_AUDIO_DIR) if f.endswith(".wav")]) if os.path.exists(YOUTUBE_AUDIO_DIR) else []
        jobs = [(os.path.join(YOUTUBE_AUDIO_DIR, f), f"yt_{f[:-4]}.pt", yt_metadata[f[:-4]], {})
                for f in yt_wav_files if f[:-4] in yt_metadata]
        print(f"\n🎥 Processing {len(jobs)} Real-World YouTube Audio Files from '{YOUTUBE_AUDIO_DIR}'...")
        summary["YouTube"] = process(jobs, "YouTube Audio")
        
    print("\n" + "=" * 75)
    print(f"🎉 Unified Feature Extraction Complete!")
    for name, (processed, skipped) in summary.items():
        print(f"  • {name + ' Dialogues:':<25} {processed} processed, {skipped} skipped (up to date)")
    print(f"  • Tensors saved in:         '{OUTPUT_DIR}'")
    if failed:
        print(f"  • ❌ {len(failed)} files failed (re-run with --skip-existing to retry only these):")
        for path in failed:
            print(f"      {path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
