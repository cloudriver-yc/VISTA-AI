"""
Acoustic data augmentation for the synthetic TTS dialogues.

Takes the stereo ElevenLabs clips in data/synthetic/tts/ (customer on ch0, engineer on ch1),
mixes them down to mono like a real phone recording, and applies a random chain of
speed / pitch / reverb / background noise / telephone channel / gain perturbations.

Output: data/synthetic/augmented/{source_id}__aug{NN}.flac + manifest.jsonl (labels + exact params).
The number of variants is class-balanced and spread evenly across unique scripts, so
re-voiced script copies (dial_aug_*) do not dominate the augmented pool.
"""
import os
import io
import sys
import csv
import json
import glob
import zipfile
import hashlib
import argparse
import urllib.request
from collections import defaultdict

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt, fftconvolve
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

SR = 16000
AUDIO_DIR = paths.TTS_DIR
TURN_CACHE_DIR = paths.TTS_TURN_CACHE_DIR
JSON_PATH = paths.DIALOGUES_PATH
OUTPUT_DIR = paths.AUGMENTED_DIR
PREVIEW_DIR = paths.AUGMENTED_PREVIEW_DIR
NOISE_DIR = paths.NOISE_DIR
ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
CLASSES = ["very_unsatisfied", "unsatisfied", "satisfied", "very_satisfied"]

# ESC-50 targets 20-29 are human non-speech sounds (laughing, crying, coughing, ...).
# They are excluded so background noise never carries an emotional cue.
ESC50_EXCLUDED_TARGETS = set(range(20, 30))


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def text_key(dialogue):
    """Hash of the turn texts; two rows with the same key are the same script."""
    joined = "||".join(t["text"].strip().lower() for t in dialogue["turns"])
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def load_sources():
    """Returns {label: [[source_id, ...] per unique script]} for every voiced dialogue."""
    by_class = defaultdict(lambda: defaultdict(list))
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d["action_label"] not in CLASSES:
                continue
            if os.path.exists(os.path.join(AUDIO_DIR, f"{d['dialogue_id']}.wav")):
                by_class[d["action_label"]][text_key(d)].append(d["dialogue_id"])
    return {label: [sorted(ids) for _, ids in sorted(groups.items())] for label, groups in by_class.items()}


def build_jobs(sources, target, seed):
    """Class-balanced list of (label, source_id, aug_index, variant_seed), round-robin over scripts."""
    jobs = []
    per_class = [target // len(CLASSES) + (1 if i < target % len(CLASSES) else 0) for i in range(len(CLASSES))]
    for class_idx, label in enumerate(CLASSES):
        groups = sources.get(label, [])
        if not groups:
            print(f"⚠️ No voiced sources for class '{label}', skipping.")
            continue
        aug_counter = defaultdict(int)
        for k in range(per_class[class_idx]):
            group = groups[k % len(groups)]
            source_id = group[(k // len(groups)) % len(group)]
            idx = aug_counter[source_id]
            aug_counter[source_id] += 1
            variant_seed = int(hashlib.sha1(f"{seed}:{source_id}:{idx}".encode()).hexdigest()[:8], 16)
            jobs.append((label, source_id, idx, variant_seed))
    return jobs


# ---------------------------------------------------------------------------
# Noise banks
# ---------------------------------------------------------------------------
def download_esc50():
    if os.path.exists(os.path.join(NOISE_DIR, "meta", "esc50.csv")):
        print(f"✅ ESC-50 already present in '{NOISE_DIR}'.")
        return
    print(f"📥 Downloading ESC-50 (~600 MB) from {ESC50_URL} ...")
    with urllib.request.urlopen(ESC50_URL) as resp:
        payload = resp.read()
    os.makedirs(NOISE_DIR, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for member in zf.namelist():
            rel = member.split("/", 1)[1] if "/" in member else ""
            if not (rel.startswith("audio/") or rel == "meta/esc50.csv") or member.endswith("/"):
                continue
            dest = os.path.join(NOISE_DIR, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())
    print(f"✅ ESC-50 extracted to '{NOISE_DIR}'.")


def list_esc50_files():
    meta = os.path.join(NOISE_DIR, "meta", "esc50.csv")
    if not os.path.exists(meta):
        return []
    with open(meta, newline="") as f:
        rows = list(csv.DictReader(f))
    return sorted(os.path.join(NOISE_DIR, "audio", r["filename"]) for r in rows
                  if int(r["target"]) not in ESC50_EXCLUDED_TARGETS)


def list_turn_clips():
    """Cached per-turn TTS clips, keyed by the dialogue they belong to (used to build babble)."""
    clips = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(TURN_CACHE_DIR, "*_turn_*.npy"))):
        dialogue_id = os.path.basename(path).rsplit("_turn_", 1)[0]
        clips[dialogue_id].append(path)
    return clips


# ---------------------------------------------------------------------------
# DSP helpers
# ---------------------------------------------------------------------------
def active_rms(y, frame=400):
    """RMS over voiced frames only, so long TTS silences don't skew SNR."""
    n = len(y) // frame
    if n == 0:
        return float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    energies = np.mean(y[:n * frame].reshape(n, frame) ** 2, axis=1)
    active = energies[energies > 0.05 * energies.max()]
    return float(np.sqrt(np.mean(active)) + 1e-9)


def fit_length(noise, length, rng):
    if len(noise) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(noise) < length:
        noise = np.tile(noise, int(np.ceil(length / len(noise))))
    start = rng.integers(0, len(noise) - length + 1)
    return noise[start:start + length]


def colored_noise(length, color, rng):
    exponent = {"white": 0.0, "pink": 1.0, "brown": 2.0}[color]
    spectrum = rng.standard_normal(length // 2 + 1) + 1j * rng.standard_normal(length // 2 + 1)
    freqs = np.fft.rfftfreq(length)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    noise = np.fft.irfft(spectrum / freqs ** (exponent / 2.0), n=length)
    return noise.astype(np.float32)


def hum_noise(length, mains_hz, rng):
    t = np.arange(length) / SR
    phase = rng.uniform(0, 2 * np.pi, size=4)
    hum = np.zeros(length)
    for k in range(4):
        hum += (0.6 ** k) * np.sin(2 * np.pi * mains_hz * (k + 1) * t + phase[k])
    return hum.astype(np.float32)


def babble_noise(length, n_talkers, exclude_id, turn_clips, rng):
    candidates = [p for did, paths in turn_clips.items() if did != exclude_id for p in paths]
    babble = np.zeros(length, dtype=np.float32)
    if not candidates:
        return babble
    for path in rng.choice(candidates, size=min(n_talkers, len(candidates)), replace=False):
        clip = np.load(path).astype(np.float32) / 32768.0
        babble += fit_length(clip, length, rng)
    return babble


def synthetic_rir(rt60, rng):
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    rir[0] = 1.0
    return (rir / np.max(np.abs(rir))).astype(np.float32)


def mu_law(y, mu=255):
    y = np.clip(y, -1.0, 1.0)
    encoded = np.round((np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu) + 1) / 2 * mu)
    decoded = 2 * encoded / mu - 1
    return (np.sign(decoded) * ((1 + mu) ** np.abs(decoded) - 1) / mu).astype(np.float32)


def db_to_gain(db):
    return 10.0 ** (db / 20.0)


# ---------------------------------------------------------------------------
# Augmentation chain
# ---------------------------------------------------------------------------
def sample_params(rng, channel_mode, esc50_files, has_babble):
    """Draws the full augmentation recipe for one variant (recorded in the manifest)."""
    p: dict = {"channel": channel_mode}
    if channel_mode == "mix":
        p["customer_gain_db"] = round(float(rng.uniform(-6, 6)), 2)
        p["engineer_gain_db"] = round(float(rng.uniform(-6, 6)), 2)
    if rng.random() < 0.3:
        p["speed"] = round(float(rng.uniform(0.9, 1.1)), 3)
    if rng.random() < 0.3:
        p["pitch_semitones"] = round(float(rng.uniform(-2, 2)), 2)
    if rng.random() < 0.4:
        p["reverb_rt60"] = round(float(rng.uniform(0.2, 0.8)), 2)

    noise_types = ["colored", "hum"] + (["esc50"] * 2 if esc50_files else []) + (["babble"] if has_babble else [])
    noise_type = str(rng.choice(noise_types))
    noise: dict = {"type": noise_type}
    p["noise"] = noise
    if noise_type == "colored":
        noise["color"] = str(rng.choice(["white", "pink", "brown"]))
        noise["snr_db"] = round(float(rng.uniform(5, 25)), 1)
    elif noise_type == "hum":
        noise["mains_hz"] = int(rng.choice([50, 60]))
        noise["snr_db"] = round(float(rng.uniform(15, 30)), 1)
    elif noise_type == "esc50":
        noise["file"] = os.path.basename(str(rng.choice(esc50_files)))
        noise["snr_db"] = round(float(rng.uniform(5, 25)), 1)
    else:
        noise["talkers"] = int(rng.integers(3, 7))
        # Babble is kept quieter so Whisper doesn't transcribe the background talkers
        noise["snr_db"] = round(float(rng.uniform(12, 25)), 1)

    if rng.random() < 0.5:
        p["telephone"] = {"mu_law": bool(rng.random() < 0.5)}
    p["final_gain_db"] = round(float(rng.uniform(-6, 6)), 2)
    return p


def augment(stereo, params, rng, source_id, turn_clips):
    # 1. Channel handling: mono mixdown like a real phone recording, or customer channel only
    if stereo.ndim == 1:
        y = stereo.copy()
    elif params["channel"] == "mix":
        y = stereo[:, 0] * db_to_gain(params["customer_gain_db"]) + stereo[:, 1] * db_to_gain(params["engineer_gain_db"])
    else:
        y = stereo[:, 0].copy()
    y = y.astype(np.float32)

    # 2. Speed perturbation (changes tempo and pitch, like Kaldi speed perturb)
    if "speed" in params:
        y = librosa.resample(y, orig_sr=SR, target_sr=int(SR / params["speed"]))
    # 3. Pitch shift (tempo preserved)
    if "pitch_semitones" in params:
        y = librosa.effects.pitch_shift(y, sr=SR, n_steps=params["pitch_semitones"])
    # 4. Room reverb
    if "reverb_rt60" in params:
        dry_rms = active_rms(y)
        y = fftconvolve(y, synthetic_rir(params["reverb_rt60"], rng))[:len(y)].astype(np.float32)
        y *= dry_rms / active_rms(y)

    # 5. Background noise at the requested SNR
    noise_cfg = params["noise"]
    if noise_cfg["type"] == "colored":
        noise = colored_noise(len(y), noise_cfg["color"], rng)
    elif noise_cfg["type"] == "hum":
        noise = hum_noise(len(y), noise_cfg["mains_hz"], rng)
    elif noise_cfg["type"] == "esc50":
        clip, clip_sr = sf.read(os.path.join(NOISE_DIR, "audio", noise_cfg["file"]), dtype="float32")
        if clip.ndim > 1:
            clip = clip.mean(axis=1)
        noise = fit_length(librosa.resample(clip, orig_sr=clip_sr, target_sr=SR), len(y), rng)
    else:
        noise = babble_noise(len(y), noise_cfg["talkers"], source_id, turn_clips, rng)
    noise_rms = float(np.sqrt(np.mean(noise ** 2)) + 1e-9)
    y = y + noise * (active_rms(y) / noise_rms) * db_to_gain(-noise_cfg["snr_db"])

    # 6. Telephone channel: 8 kHz narrowband, 300-3400 Hz band-pass, optional G.711 mu-law
    if "telephone" in params:
        y = librosa.resample(y, orig_sr=SR, target_sr=8000)
        y = np.asarray(sosfilt(butter(4, [300, 3400], btype="bandpass", fs=8000, output="sos"), y), dtype=np.float32)
        if params["telephone"]["mu_law"]:
            y = mu_law(y)
        y = librosa.resample(y, orig_sr=8000, target_sr=SR)

    # 7. Final gain with a peak guard against clipping
    y = y * db_to_gain(params["final_gain_db"])
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Augment synthetic TTS dialogues with noise, reverb, telephone and speed/pitch perturbations")
    parser.add_argument("--target", type=int, default=2500, help="Total augmented clips to produce (class-balanced)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic augmentation recipes")
    parser.add_argument("--channel", choices=["mix", "customer"], default="mix",
                        help="'mix' = mono mixdown of both speakers (matches real calls); 'customer' = ch0 only")
    parser.add_argument("--preview", type=int, default=0,
                        help="Write only N sample clips as WAV to data/synthetic/augmented_preview/ for listening")
    parser.add_argument("--download-noise", action="store_true", help="Download ESC-50 background noise before augmenting")
    args = parser.parse_args()

    if args.download_noise:
        download_esc50()

    esc50_files = list_esc50_files()
    turn_clips = list_turn_clips()
    if not esc50_files:
        print("⚠️ ESC-50 not found; using procedural noise only (run with --download-noise to add it).")

    sources = load_sources()
    jobs = build_jobs(sources, args.target, args.seed)
    print(f"📂 {sum(len(ids) for g in sources.values() for ids in g)} voiced sources across "
          f"{sum(len(g) for g in sources.values())} unique scripts -> {len(jobs)} augmented clips")

    if args.preview:
        # Spread the preview across classes and sources
        stride = max(1, len(jobs) // args.preview)
        jobs = jobs[::stride][:args.preview]
        out_dir, ext = PREVIEW_DIR, "wav"
    else:
        out_dir, ext = OUTPUT_DIR, "flac"
    os.makedirs(out_dir, exist_ok=True)

    manifest = []
    n_written = 0
    for label, source_id, idx, variant_seed in tqdm(jobs, desc="Augmenting"):
        rng = np.random.default_rng(variant_seed)
        params = sample_params(rng, args.channel, esc50_files, bool(turn_clips))
        filename = f"{source_id}__aug{idx:02d}.{ext}"
        out_path = os.path.join(out_dir, filename)

        if not os.path.exists(out_path):
            stereo, sr = sf.read(os.path.join(AUDIO_DIR, f"{source_id}.wav"), dtype="float32")
            if sr != SR:
                stereo = librosa.resample(stereo.T, orig_sr=sr, target_sr=SR).T
            y = augment(stereo, params, rng, source_id, turn_clips)
            sf.write(out_path, y, SR)
            n_written += 1

        manifest.append({"file": filename, "source_id": source_id, "label": label, "seed": variant_seed, "params": params})

    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for rec in manifest:
            f.write(json.dumps(rec) + "\n")

    counts = defaultdict(int)
    for rec in manifest:
        counts[rec["label"]] += 1
    print("\n" + "=" * 75)
    print(f"🎉 Augmentation complete: {n_written} new clips written, {len(manifest) - n_written} already existed.")
    for label in CLASSES:
        print(f"  • {label:<17} {counts[label]}")
    print(f"  • Output:   '{out_dir}'")
    print(f"  • Manifest: '{manifest_path}'")
    print("=" * 75)


if __name__ == "__main__":
    main()
