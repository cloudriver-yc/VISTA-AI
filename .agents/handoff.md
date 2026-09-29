# VISTA-AI: Project Handoff & Architecture Guide

Welcome to the VISTA-AI repository! This document serves as the master blueprint for the project to help human collaborators and future AI agents immediately understand the architecture, the directory structure, and the sequential execution workflows.

## 1. Project Overview
VISTA-AI is a multimodal PyTorch deep learning framework designed to predict Customer Satisfaction (CSAT) scores by combining **linguistic text semantics** and **acoustic prosody features** directly from conversational audio. 

The architecture supports dual comparative implementations:
- **Linguistic Path (Text):** 768-dimensional sentence embeddings (via `sentence-transformers/all-mpnet-base-v2`).
- **Acoustic Path (Audio):** 768-dimensional self-supervised speech representations extracted directly from raw 16kHz audio chunks via **`microsoft/wavlm-base-plus`** (capturing fine-grained pitch inflection, vocal tension, sarcasm, and prosody).
- **V1 Baseline Architecture (`DualTransformerClassifier`):**
  - Linear Projections with LayerNorm, GELU, and Dropout.
  - Bidirectional Cross-Modal Attention ($Q=T, K=A, V=A$ and $Q=A, K=T, V=T$).
  - 2-layer Sequence Transformer Encoder across dialogue segments with Sinusoidal Positional Encoding (up to 1024 turns).
  - Mean-pooled 4-class CSAT classification head.
  - Saved weights: `models/dual_transformer_v1_weights.pt`.
- **V2 Upgraded Architecture (`EnhancedDualTransformerClassifier`):**
  - Pre-LN 2-layer **`MLPResBlock`** Feature Adaptation Head for non-linear domain tuning.
  - Bidirectional Cross-Modal Attention Fusion.
  - Learnable **`[CLS]` Token** prepending to explicitly model global conversational sentiment.
  - Saved weights: `models/dual_transformer_v2_weights.pt` (and `models/dual_transformer_weights.pt`).
- **Hardware Acceleration:** Native Apple Silicon GPU acceleration via PyTorch MPS backend (`torch.device("mps")`), falling back to `cuda` then `cpu` on other machines.
- **Multimodal Input Ingestion (`app.py`):** the Streamlit dashboard accepts three input sources — file upload (`mp3`/`wav`/`m4a`/`flac`/`aac`/`opus`/`ogg`/`mp4`/`webm`/`mov`), live browser microphone recording (`st.audio_input`), or a YouTube URL — all normalized to 16kHz mono WAV via `ffmpeg` before running the same Whisper → WavLM → MPNet → V1/V2 pipeline.
- **Speaker Diarization (heuristic):** dialogue segments are grouped into `Speaker 1` / `Speaker 2` / etc. by clustering the per-segment WavLM embeddings (KMeans, auto-selecting cluster count via silhouette score). This reuses embeddings already computed for CSAT inference — no separate diarization model or extra dependency — so it's an approximate speaker-turn heuristic, not verified speaker identity.

## 2. Directory Structure
The repository has been structured according to strict software engineering standards:

```text
VISTA-AI/
├── .agents/                    # Workspace rules and handoff context
├── .env                        # Environment variables (e.g., ELEVENLABS_API_KEY, HF_TOKEN)
├── CLAUDE.md                   # Guidance for Claude Code: commands, architecture, pitfalls (label mapping, split manifest)
├── app.py                      # Interactive Streamlit Web UI (Side-by-side V1 vs V2)
├── data/                       # Datasets & metadata, grouped by provenance (all paths defined in src/paths.py)
│   ├── synthetic/              # TRAINING pool
│   │   ├── dialogues.jsonl     #   Generated dialogue scripts (labels, turns, voice profiles, generator model)
│   │   ├── tts/                #   ElevenLabs stereo WAVs (customer ch0, engineer ch1)
│   │   ├── tts_turn_cache/     #   Per-turn TTS cache (*_turn_N.npy), also used to build babble noise
│   │   ├── augmented/          #   Augmented mono FLAC clips + manifest.jsonl (exact recipe per clip)
│   │   └── augmented_preview/  #   Small WAV batch for listening QA (--preview N)
│   ├── real/                   # TEST pool (never trained on)
│   │   ├── youtube/            #   audio/ (16 kHz WAV), captions/ (VTT), metadata.jsonl
│   │   └── uploads/            #   audio/ (<id>.mp3), metadata.jsonl (upload_id, label)
│   ├── noise/esc50/            # ESC-50 background noise bank (downloaded, CC BY-NC)
│   ├── features/               # One .pt per dialogue: dial* (synthetic), yt_* / upload_* (real)
│   ├── split_manifest.json     # train / val (synthetic) · finetune (uploads) · test (YouTube) file lists
│   └── tmp/                    # Scratch for app.py / CLI downloads (safe to delete)
├── models/                     # Stage-1 weights (dual_transformer_v{1,2}_weights.pt) + stage-2 adapters (csat_adapter_v{1,2}.pt)
├── scripts/                    # Pipeline Scripts
│   ├── 01_generate_scripts.py     # Generates NEW unique scripts via Antigravity CLI (--per-class target, --long-frac, --models)
│   ├── 03_synthesize_audio.py     # Converts JSON transcripts to TTS audio via ElevenLabs (--only-missing)
│   ├── 03b_augment_audio.py       # Acoustic augmentation: mono mixdown, noise, reverb, telephone, speed/pitch
│   ├── 04_extract_features.py     # Whisper → WavLM + MPNet features (--only syn,aug,yt --skip-existing)
│   ├── 05_test_youtube.py         # Side-by-side V1 vs V2 validation script with telemetry
│   ├── evaluate_asr.py            # Evaluates Whisper ASR accuracy vs YouTube captions (WER)
│   ├── ingest_youtube_dataset.py  # Ingests YouTube playlists, removes B-roll & extracts features
│   ├── ingest_upload.py           # Adds a labelled real uploaded call (audio -> features + metadata) for stage 2
│   ├── migrate_data_layout.py     # Moves the legacy data/ layout to the current one (idempotent)
│   └── hf_dataset_sync.py         # Hugging Face sync by part (metadata/features/models/synthetic-audio/real-audio); mirrors features + models
└── src/                        # Core PyTorch Framework
    ├── paths.py                # Single source of truth for every data/model path
    ├── audio_utils.py          # segment_audio(): slices Whisper segments, widening <25 ms ones to WavLM's minimum
    ├── feature_extraction.py   # Whisper → WavLM + MPNet extraction shared by 04_extract_features.py and ingest_upload.py
    ├── data/
    │   └── dataset.py          # RealCSATDataset + script-grouped train/val/test split manifest
    ├── models/
    │   ├── architecture.py     # DualTransformerClassifier (V1), EnhancedDualTransformerClassifier (V2), MLPResBlock
    │   └── explainer.py        # MultimodalExplainer (XAI: turn saliency, modality ratio, tone diagnosis, narrative)
    └── train.py                # Two-stage training: --stage pretrain | finetune | all, then YouTube test
```

## 3. The Execution Workflow

### Step 1: Download or Ingest Data
- **Hugging Face Sync:**
  ```bash
  python scripts/hf_dataset_sync.py download                                   # everything (~8 GB)
  python scripts/hf_dataset_sync.py download --parts metadata features models  # ~600 MB: enough to train and run the app
  python scripts/hf_dataset_sync.py status --remote                            # compare local vs Hub per part
  python scripts/hf_dataset_sync.py upload --dry-run                           # preview, then run without --dry-run
  ```
  Uploads mirror `features/` and `models/` exactly (stale feature files would otherwise enter training via `--rebuild_split`, and a stale adapter would be loaded by the app). Legacy-layout files (`data/audio/`, `data/raw/`, …) are pruned only when their new-layout copy is uploaded in the same run. Legacy files with no local copy are kept unless `--prune-orphans`. `data/tmp/`, `data/noise/` and `augmented_preview/` are never synced.
- **Ingest Real YouTube Playlists:**
  ```bash
  python scripts/ingest_youtube_dataset.py
  ```

### Step 2: Synthetic Data Generation & Acoustic Augmentation
New unique scripts (Antigravity CLI `agy`; Gemini 3.8 Flash Medium first, falling back to 3.8 Flash High / 3.1 Pro / Claude Sonnet 4.6 on quota) → stereo TTS (ElevenLabs) → augmented mono clips:
```bash
python scripts/01_generate_scripts.py --per-class 60 --long-frac 0.4
python scripts/03_synthesize_audio.py --only-missing
python scripts/03b_augment_audio.py --download-noise --target 2500
```
Scripts are a mix of short calls (20–60 s, 4–10 turns) and, with `--long-frac`, long calls (2–5 min, 20–40 turns with a verification → troubleshooting → hold/escalation → outcome arc). Real test calls have a median length of ~4 min and 46–105 Whisper segments, versus ~30 s and 5 segments for the original TTS scripts.

Each augmented clip gets a random, recorded recipe (`data/synthetic/augmented/manifest.jsonl`):
- **Mono mixdown** of customer + engineer channels (±6 dB per speaker), matching real phone recordings and the app's ffmpeg mono input.
- **Speed** 0.9–1.1× (p=0.3), **pitch** ±2 semitones (p=0.3), **room reverb** RT60 0.2–0.8 s (p=0.4).
- **One background noise** source: ESC-50 environmental sounds (human-vocal classes excluded), white/pink/brown noise, 50/60 Hz hum, or call-centre babble built from other TTS turns, at 5–30 dB SNR.
- **Telephone channel** (p=0.5): 8 kHz narrowband, 300–3400 Hz band-pass, optional G.711 μ-law.
- **Final gain** ±6 dB with a peak guard.

Clips are class-balanced and spread round-robin over *unique scripts*, so re-voiced copies (`dial_aug_*`) don't dominate. `--target` scales the set (e.g. 25000).

### Step 3: Feature Extraction (Whisper ASR + WavLM + SentenceTransformer)
Extract multimodal segment representations (stereo synthetic audio is summed to mono by default, `--channel customer` restores the old ch0-only behaviour):
```bash
python scripts/04_extract_features.py --only syn,aug --skip-existing
```

### Step 4: Two-Stage Training of V1 and V2 (professor's transfer-learning recipe)
```bash
python src/train.py --stage all --rebuild_split
```
1. **Stage 1: pretrain on synthetic data** (`--stage pretrain`). V1/V2 train on the synthetic train split; each epoch reports train and synthetic-val accuracy, and the best-val checkpoint is saved to `models/dual_transformer_v{1,2}_weights.pt`.
2. **Stage 2: freeze + new layers on uploaded calls** (`--stage finetune`). The stage-1 model is frozen and a `ResidualAdapterHead` (LayerNorm → Linear 512→64 → GELU → Dropout → Linear 64→4) is added on its 512-d call summary (V1 pooled vector / V2 `[CLS]`). Its output is added to the frozen logits, and its last layer is zero-initialised, so before training it predicts exactly what stage 1 predicts. Only the adapter trains, on the uploaded calls (`upload_*`, full batch, `--ft_epochs 30`, `--ft_lr 1e-3`). The script asserts the frozen weights are unchanged and saves `models/csat_adapter_v{1,2}.pt`. Two safeguards stop the adapter from overfitting a handful of uploads (without them it collapsed V2 to predicting "Unsatisfied" for 10 of 19 YouTube calls and cut its synthetic-val accuracy from 99.4% to 73.1%): an **anchor** KL penalty (`--ft_anchor_weight 1.0`) keeps the adapted predictions close to the frozen model's on 512 synthetic training calls, and a **guard** rejects the adapter (falling back to stage 1) if it lowers synthetic-val accuracy by more than `--ft_max_val_drop 1.0` points. Neither looks at YouTube, so the test set stays clean.
3. **Stage 3: test on YouTube.** Every YouTube call is scored by both the stage-1 model and the stage-2 model, and the final table shows `YT stage1`, `YT stage2` and the difference.

`app.py` and `05_test_youtube.py` automatically use the adapter when `models/csat_adapter_v{1,2}.pt` exists ("+ adapter" in the UI). Re-running `--stage pretrain` alone deletes old adapters, since they belong to the previous stage-1 model.

**Adding uploaded calls** (stage-2 data), then retraining only the adapter:
```bash
python scripts/ingest_upload.py --audio ~/Downloads/call.mp3 --label unsatisfied
python src/train.py --stage finetune --rebuild_split
```

### Step 5: Side-by-Side Inference & Telemetry
- **CLI Comparative Test with Telemetry:**
  ```bash
  python scripts/05_test_youtube.py
  ```
- **Streamlit Web Dashboard:**
  ```bash
  streamlit run app.py
  ```
  Accepts an uploaded audio file, a live mic recording, or a YouTube URL; plays back the analyzed clip inline; shows live elapsed-time progress during transcription; and labels transcript turns by speaker (heuristic clustering, see above).

## 4. Current Status & Verification
- **4 CSAT Categories:**
  1. `Very Unsatisfied` (Happy / strong emotion, problem solved) — `_very_unsatisfied.wav`
  2. `Unsatisfied` (Flat emotion, problem not solved) — `_unsatisfied.wav`
  3. `Satisfied` (Flat emotion, problem solved) — `_satisfied.wav`
  4. `Very Satisfied` (Strong angry / shouting, problem not solved) — `_very_satisfied.wav`
- **Dataset Size (before the augmentation rollout):** 307 synthetic TTS clips in `data/synthetic/tts/` built from only **16 unique scripts** (16 base `dial_NNN` + 291 `dial_aug_*` re-voicings with identical text), all ~30 s / ~5 segments; 19 real-world YouTube dialogues and 2 uploaded real calls. `dialogues.jsonl` holds 506 rows / 22 unique scripts. *Update with the new counts after running the Augment & Retrain steps below.*
- **Train / Val / Finetune / Test (`data/split_manifest.json`):**
  - **Train:** synthetic TTS clips + their augmented variants only (stage 1).
  - **Val:** ~15% of the unique *scripts* per class (with all their re-voicings and augmented variants), synthetic only (stage-1 model selection).
  - **Finetune:** the uploaded real calls, currently 2 (1 Unsatisfied, 1 Satisfied); stage 2 trains only the adapter on these.
  - **Test:** the 19 YouTube calls (3 / 2 / 9 / 5 per class), never trained on (stage 3).
  - **YouTube label fix (2026-09-28):** the YouTube playlists were labelled with the *everyday* meaning of the two extreme classes (angry calls under "very_unsatisfied"), the reverse of the project mapping. The 8 affected `yt_*.pt` files were swapped 0 ↔ 3 (marked `label_convention: "project"`; originals in `data/real/youtube/features_before_label_swap/`), and `ingest_youtube_dataset.py` now assigns the swapped labels.
  - **Zero Leakage:** splits are grouped by a hash of the script text, so no dialogue text is shared between train and val.
  - A legacy `{"train","test"}` manifest is backed up to `split_manifest.v1.json` and regenerated automatically.
- **Benchmark Accuracies (MPS GPU, two-stage recipe, 2026-09-28):** 2,569 synthetic train / 484 synthetic val, 2 uploads for stage 2, 19 YouTube test calls.
  - V1 Baseline (Mean Pooling): **97.7% synthetic val**, **78.9% YouTube (stage 1)**, **78.9% YouTube (stage 2)**; the adapter was accepted but net-neutral (fixes one call, breaks another).
  - V2 Upgraded (`MLPResBlock` + `[CLS]`): **99.4% synthetic val**, **78.9% YouTube (stage 1)**; the stage-2 adapter was rejected by the guard (synthetic val −2.5 pts), so stage 2 = stage 1.
  - Remaining YouTube errors are concentrated in 4 calls: two Satisfied calls predicted Very Unsatisfied, one Unsatisfied predicted Satisfied, and one angry dispute (`yt_gD7xQGXpSBg`, Very Satisfied) predicted Unsatisfied.
  - Earlier numbers (40% / 20% YouTube, then 47.4% / 42.1%) were measured against the unswapped YouTube labels and are not comparable.

## 5. Augment & Retrain (step list)
Run from the repo root with the `.venv` active.
```bash
source .venv/bin/activate && pip install -r requirements.txt
# 0. Back up
cp data/synthetic/dialogues.jsonl data/synthetic/dialogues.v1.jsonl
cp -r models models_v1_backup
# 1. Write ~240 new unique scripts, 40% long calls (Antigravity CLI, your Google account; ~2 h).
#    If a quota stops it, re-run the same command later: it only fills what is missing.
python scripts/01_generate_scripts.py --per-class 60 --long-frac 0.4
# 2. Voice only new / unvoiced unique scripts (ElevenLabs, ~300k characters)
python scripts/03_synthesize_audio.py --only-missing
# 3. Download noise + listen to a preview, then augment
python scripts/03b_augment_audio.py --download-noise --preview 40
python scripts/03b_augment_audio.py --target 2500
# 4. Extract features (~4 s per short clip, ~30 s per long clip on M1 Pro: roughly 10 h total, less on newer chips)
python scripts/04_extract_features.py --only syn,aug --skip-existing
# 5. Two-stage training: pretrain on synthetic -> frozen model + adapter on uploads -> test on YouTube
python src/train.py --stage all --rebuild_split
#    Later, after adding uploads with scripts/ingest_upload.py, retrain only the adapter:
#    python src/train.py --stage finetune --rebuild_split
# 6. Check
python scripts/05_test_youtube.py
streamlit run app.py
```
