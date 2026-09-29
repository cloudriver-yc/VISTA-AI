"""
Adds a real uploaded call to the stage-2 (finetune) pool.

  python scripts/ingest_upload.py --audio ~/Downloads/call.mp3 --label unsatisfied

1. Copies the audio to data/real/uploads/audio/<id>.<ext>
2. Converts it to 16 kHz mono WAV with ffmpeg (same as app.py)
3. Extracts Whisper + WavLM + MPNet features to data/features/upload_<id>.pt
4. Records it in data/real/uploads/metadata.jsonl

Then retrain only the adapter: python src/train.py --stage finetune --rebuild_split
"""
import os
import sys
import json
import shutil
import argparse
import subprocess
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths
from feature_extraction import LABEL_MAP, extract_features_from_audio, get_device, load_extractors

CANONICAL_LABELS = ["very_unsatisfied", "unsatisfied", "satisfied", "very_satisfied"]


def main():
    parser = argparse.ArgumentParser(description="Add a labelled real uploaded call to the stage-2 finetune pool")
    parser.add_argument("--audio", required=True, help="Path to the call recording (mp3, wav, m4a, ...)")
    parser.add_argument("--label", required=True, choices=CANONICAL_LABELS,
                        help="CSAT label (non-intuitive mapping: very_unsatisfied = delighted & resolved, "
                             "very_satisfied = angry & unresolved; see CLAUDE.md)")
    parser.add_argument("--id", help="Upload id (default: the file name, with '.' replaced by '_')")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing upload with the same id")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        sys.exit(f"Audio file not found: {args.audio}")
    stem, ext = os.path.splitext(os.path.basename(args.audio))
    upload_id = (args.id or stem).replace(".", "_").replace(" ", "_")
    audio_dest = os.path.join(paths.UPLOADS_AUDIO_DIR, f"{upload_id}{ext.lower()}")
    feature_file = f"upload_{upload_id}.pt"
    feature_path = os.path.join(paths.FEATURES_DIR, feature_file)
    if os.path.exists(feature_path) and not args.force:
        sys.exit(f"{feature_path} already exists. Use --force to overwrite, or --id to pick another id.")

    # 1. Keep a copy of the original recording
    os.makedirs(paths.UPLOADS_AUDIO_DIR, exist_ok=True)
    if os.path.abspath(args.audio) != os.path.abspath(audio_dest):
        shutil.copy2(args.audio, audio_dest)

    # 2. Normalize to 16 kHz mono WAV, exactly like app.py does for uploads
    os.makedirs(paths.TMP_DIR, exist_ok=True)
    wav_path = os.path.join(paths.TMP_DIR, f"upload_{upload_id}.wav")
    subprocess.run(["ffmpeg", "-y", "-i", audio_dest, "-ar", "16000", "-ac", "1", wav_path],
                   check=True, capture_output=True)

    # 3. Extract features
    extractors = load_extractors(get_device())
    audio_emb, text_emb = extract_features_from_audio(wav_path, **extractors)
    label = LABEL_MAP[args.label]
    os.makedirs(paths.FEATURES_DIR, exist_ok=True)
    torch.save({
        "audio_embeds": audio_emb,
        "text_embeds": text_emb,
        "label": torch.tensor(label, dtype=torch.long),
        "channel": "mix",
        "source_id": f"upload_{upload_id}",
    }, feature_path)

    # 4. Record it in the uploads metadata (replacing any earlier row for the same id)
    rows = []
    if os.path.exists(paths.UPLOADS_METADATA_PATH):
        with open(paths.UPLOADS_METADATA_PATH, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    rows = [r for r in rows if r.get("upload_id") != upload_id]
    rows.append({"upload_id": upload_id, "audio_file": os.path.basename(audio_dest),
                 "feature_file": feature_file, "label": label, "label_name": args.label})
    with open(paths.UPLOADS_METADATA_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"✅ Added upload '{upload_id}' as {args.label} ({audio_emb.shape[0]} segments) -> {feature_path}")
    print(f"   {len(rows)} uploaded calls in total. Retrain the adapter with:")
    print("   python src/train.py --stage finetune --rebuild_split")


if __name__ == "__main__":
    main()
