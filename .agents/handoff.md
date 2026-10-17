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
- **Multimodal Input Ingestion (`app.py`):** the Streamlit dashboard accepts three input sources — file upload (`mp3`/`wav`/`m4a`/`flac`/`aac`/`opus`/`ogg`/`mp4`/`webm`/`mov`), live browser microphone recording (`st.audio_input`), or a YouTube URL — all normalized to 16kHz mono WAV via `ffmpeg` before running the same Whisper → WavLM → MPNet → V1/V2 pipeline. YouTube audio is fetched with `python -m yt_dlp` from the app's own interpreter (works without an activated `.venv`). On failure the app shows yt-dlp's actual error. It only suggests a manual `yt-dlp --cookies-from-browser chrome …` download + upload when YouTube really rate-limits or bot-checks (HTTP 429 / "Sign in to confirm"); a missing `ffmpeg` gets its own message.
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
│   ├── real/                   # Real calls (split into finetune / test, see data/features/)
│   │   ├── youtube/            #   audio/ (16 kHz WAV), captions/ (VTT), metadata.jsonl,
│   │   │                       #   augmented/ ({video_id}__augNN.flac + manifest.jsonl)
│   │   └── uploads/            #   audio/ (<id>.mp3), metadata.jsonl (upload_id, label),
│   │                           #   augmented/ ({upload_id}__augNN.flac + manifest.jsonl)
│   ├── noise/esc50/            # ESC-50 background noise bank (downloaded, CC BY-NC)
│   ├── features/               # One .pt per dialogue, in the folder of the split that uses it:
│   │   ├── train/              #   dial*    synthetic, stage-1 training
│   │   ├── val/                #   dial*    synthetic, held-out scripts (model selection)
│   │   ├── finetune/           #   real calls (upload_* / yt_*) + augmented upload copies, stage-2 adapter only
│   │   ├── test/               #   real calls (originals only), final test, never trained on
│   │   ├── unassigned/         #   new files waiting for `src/train.py --rebuild_split`
│   │   └── excluded/           #   augmented copies of test calls: never used (they would leak the test set)
│   ├── split_manifest.json     # Record of the split folders (rewritten when they change)
│   └── tmp/                    # Scratch for app.py / CLI downloads (safe to delete)
├── models/                     # Stage-1 weights (dual_transformer_v{1,2}_weights.pt) + stage-2 adapters (csat_adapter_v{1,2}.pt)
├── scripts/                    # Pipeline Scripts
│   ├── 01_generate_scripts.py     # Generates NEW unique scripts via Antigravity CLI (--per-class target, --long-frac, --models)
│   ├── 03_synthesize_audio.py     # Converts JSON transcripts to TTS audio via ElevenLabs (--only-missing)
│   ├── 03b_augment_audio.py       # Acoustic augmentation: mono mixdown, noise, reverb, telephone, speed/pitch (--uploads N / --youtube ID: real calls)
│   ├── 04_extract_features.py     # Whisper → WavLM + MPNet features (--only syn,aug,yt,upl_aug,yt_aug --skip-existing)
│   ├── ingest_youtube_dataset.py  # Ingests YouTube playlists, removes B-roll & extracts features
│   ├── ingest_upload.py           # Adds a labelled real uploaded call (audio -> features + metadata) for stage 2
│   ├── migrate_data_layout.py     # Moves the legacy data/ layout to the current one (idempotent)
│   ├── check_gpu.py               # Verifies torch / torchvision / WavLM / MPNet / Whisper / V1+V2 run on MPS (--quick)
│   └── hf_dataset_sync.py         # Hugging Face sync by part (metadata/features/models/synthetic-audio/real-audio); mirrors features + models
└── src/                        # Core PyTorch Framework
    ├── paths.py                # Single source of truth for every data/model path
    ├── audio_utils.py          # segment_audio(): slices Whisper segments, widening <25 ms ones to WavLM's minimum
    ├── feature_extraction.py   # Whisper → WavLM + MPNet extraction shared by 04_extract_features.py and ingest_upload.py
    ├── data/
    │   └── dataset.py          # RealCSATDataset + script-grouped split into data/features/<split>/ folders
    ├── models/
    │   ├── architecture.py     # DualTransformerClassifier (V1), EnhancedDualTransformerClassifier (V2), MLPResBlock
    │   └── explainer.py        # MultimodalExplainer (XAI: turn saliency, modality ratio, tone diagnosis, narrative)
    └── train.py                # Two-stage training: --stage pretrain | finetune | all, then YouTube test
```

## 3. The Execution Workflow

### Step 0: Environment Setup (macOS, Apple Silicon)
```bash
brew install python@3.12 ffmpeg deno   # ffmpeg: audio normalization + yt-dlp WAV conversion; deno: JS runtime yt-dlp needs for YouTube
/opt/homebrew/bin/python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # the PyPI torch wheel for arm64 already includes the MPS backend
python scripts/check_gpu.py            # compares MPS vs CPU per library; --quick skips the WavLM / MPNet / Whisper downloads
```
`check_gpu.py` also loads the trained V1/V2 weights and adapters, so it flags a checkpoint whose shape no longer matches the architecture (e.g. after a `d_model` change) before the app fails on it. Whisper runs on MPS but was slower than CPU on an M6 (2.3 s vs 1.4 s on a 30 s clip), so the app keeps it on CPU. scikit-learn, librosa and numba are CPU-only.

### Step 1: Download or Ingest Data
- **Hugging Face Sync:**
  ```bash
  python scripts/hf_dataset_sync.py download                                   # everything (~8 GB)
  python scripts/hf_dataset_sync.py download --parts metadata features models  # ~600 MB: enough to train and run the app
  python scripts/hf_dataset_sync.py status --remote                            # compare local vs Hub per part
  python scripts/hf_dataset_sync.py upload --dry-run                           # preview, then run without --dry-run
  ```
  Uploads mirror `features/` and `models/` exactly (stale feature files would otherwise enter training via `--rebuild_split`, and a stale adapter would be loaded by the app). Legacy-layout files (`data/audio/`, `data/raw/`, …) are pruned only when their new-layout copy is uploaded in the same run. Legacy files with no local copy are kept unless `--prune-orphans`. `data/tmp/`, `data/noise/` and `augmented_preview/` are never synced. A download that still contains flat `data/features/*.pt` files (a Hub copy from before the split folders) is sorted into the folders by the migration step that runs after it; the next `upload --parts features` replaces the flat Hub copy with the folder layout.
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
2. **Stage 2: freeze + new layers on real calls** (`--stage finetune`). The stage-1 model is frozen and a `ResidualAdapterHead` (LayerNorm → Linear 256→64 → GELU → Dropout → Linear 64→4) is added on its 256-d call summary (V1 pooled vector / V2 `[CLS]`). Its output is added to the frozen logits, and its last layer is zero-initialised, so before training it predicts exactly what stage 1 predicts. Only the adapter trains, on the files in `data/features/finetune/` (real calls + augmented upload copies, full batch, `--ft_epochs 30`, `--ft_lr 1e-3`). The script asserts the frozen weights are unchanged and saves `models/csat_adapter_v{1,2}.pt`. Two safeguards stop the adapter from overfitting a handful of uploads (without them it collapsed V2 to predicting "Unsatisfied" for 10 of 19 YouTube calls and cut its synthetic-val accuracy from 99.4% to 73.1%): an **anchor** KL penalty (`--ft_anchor_weight 1.0`) keeps the adapted predictions close to the frozen model's on 512 synthetic training calls, and a **guard** rejects the adapter (falling back to stage 1) if it lowers synthetic-val accuracy by more than `--ft_max_val_drop 1.0` points. Neither looks at the real test calls, so the test set stays clean.
3. **Stage 3: test on real calls.** Every call in `data/features/test/` (YouTube + uploads) is scored by both the stage-1 model and the stage-2 model, and the final table shows `Test st.1`, `Test st.2` and the difference.

`app.py` automatically uses the adapter when `models/csat_adapter_v{1,2}.pt` exists ("+ adapter" in the UI). Re-running `--stage pretrain` alone deletes old adapters, since they belong to the previous stage-1 model.

**Adding uploaded calls** (stage-2 data), augmenting them, then retraining only the adapter:
```bash
python scripts/ingest_upload.py --audio ~/Downloads/call.mp3 --label unsatisfied   # -> data/features/finetune/
python scripts/03b_augment_audio.py --uploads 5                 # 5 recorded-recipe copies per upload (mono, same chain as synthetic)
python scripts/03b_augment_audio.py --youtube UAQol-DvdVI --youtube-variants 30   # same for chosen YouTube calls
python scripts/04_extract_features.py --only upl_aug,yt_aug --skip-existing   # copies land next to their original (finetune/ or excluded/)
python src/train.py --stage finetune --rebuild_split            # add --resplit_real to re-split all real calls into finetune/test
```
Upload copies use the synthetic augmentation chain minus the per-speaker gains (uploads are already one mono channel); babble noise needs the synthetic TTS turn cache, so without it only ESC-50 / coloured / hum noise is used (the 30 upload copies: 17 ESC-50, 8 hum, 5 coloured). The YouTube label for `--youtube` comes from the call's feature file, since `youtube/metadata.jsonl` is currently empty.

### Step 5: Side-by-Side Inference & Telemetry
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
- **Train / Val / Finetune / Test (`data/features/<split>/`, 2026-10-16):** each feature file sits in the folder of the split that uses it, so the split is visible in the file tree; `data/split_manifest.json` is a record of the folders. New synthetic features land in `data/features/unassigned/` and are not used until `src/train.py --rebuild_split` sorts them into `train/` or `val/` (uploads go straight to `finetune/`, YouTube calls to `test/`). Moving the existing 3,074 files from the old flat `data/features/` kept every file's split; flat files from an old copy are sorted automatically.
  - **Train:** synthetic TTS clips + their augmented variants only (stage 1).
  - **Val:** ~15% of the unique *scripts* per class (with all their re-voicings and augmented variants), synthetic only (stage-1 model selection).
  - **Finetune (2026-10-17):** 12 real calls + 50 augmented copies = 62 files (per class 31 / 6 / 22 / 3): `yt_UAQol-DvdVI` + 30 copies (Very Unsatisfied, the only Very Unsatisfied finetune call); `upload_1735404531_458927` + 5 copies (Unsatisfied); `upload_InboundSampleRecording`, `upload_OUTBOUNDSAMPLE_01`, `upload_OUTBOUNDSAMPLE_02` (+ 5 copies each), `yt_DMhuP-mdCGE`, `yt_OhPJTMCT75g`, `yt_VcTNSTnPSB4`, `yt_n0FoslGKReA` (Satisfied); `yt_Ryas9OANw-E`, `yt_d-8H8tMzTo0`, `yt_yA6A-2J5hJo` (Very Satisfied). Stage 2 trains only the adapter on these.
  - **Test (2026-10-17):** 13 real calls, originals only (per class 2 / 2 / 7 / 2): `yt_DwOVXz74N9c`, `yt_QTZq96twjs0`; `yt_dGFK8vGxX7g`, `yt_xbyEs7DJshw`; `upload_1755884171_51632`, `upload_SampleInboundCall2`, `yt_4ostqJD3Psc`, `yt_LCJPL9kciOo`, `yt_SXGtFXRz2Lg`, `yt_Xi44fYgJYSU`, `yt_hpZFJctBUHQ`; `yt_RVdNobKNMig`, `yt_gD7xQGXpSBg`.
  - **How the real split was made:** `src/train.py --resplit_real` (`--real_test_ratio 0.5`, split seed 42) re-split all 25 real originals (19 YouTube + 6 uploads) per class: `round(n × 0.5)` to test, at least one call on each side. A plain `--rebuild_split` keeps real calls where they are (move a file to change its split); new uploads go to `finetune/`, new YouTube calls to `test/`.
  - **No finetune → test leakage:** augmented copies (`upload_<id>__augNN.pt`) always follow their original. Copies of a test call go to `data/features/excluded/` (currently the 10 copies of the 2 test uploads) and are never used. Every rebuild runs a check (`_check_no_leak`) that fails if a call or any copy of it is on both sides, or a copy is in test. Before splitting, all 25 originals were compared pairwise (mean text / audio embedding cosine) to rule out duplicate recordings: the highest pair was 0.875 text / 0.922 audio, with different turn counts (topically similar calls, not the same recording).
  - Earlier (2026-10-16): 6 finetune calls (2 uploads + 1 random YouTube call per class) and 15 YouTube test calls; before that, 2 uploads / 19 YouTube.
  - **YouTube label fix (2026-09-28):** the YouTube playlists were labelled with the *everyday* meaning of the two extreme classes (angry calls under "very_unsatisfied"), the reverse of the project mapping. The 8 affected `yt_*.pt` files were swapped 0 ↔ 3 (marked `label_convention: "project"`; originals in `data/real/youtube/features_before_label_swap/`), and `ingest_youtube_dataset.py` now assigns the swapped labels.
  - **Zero Leakage:** splits are grouped by a hash of the script text, so no dialogue text is shared between train and val.
  - A legacy `{"train","test"}` manifest (backed up earlier as `split_manifest.v1.json`) is ignored; such files are re-split.
- **Benchmark Accuracies, 13-call real test, +30 Very Unsatisfied copies (2026-10-17, current models):** `yt_UAQol-DvdVI` augmented ×30 (`--youtube`), stage 2 retrained on 62 finetune files.
  - V1: adapter **accepted** (synthetic val 99.2% → 99.2%). Real test **84.6% stage 1 → 84.6% stage 2**: it no longer breaks `yt_DwOVXz74N9c`, but fixes nothing (same 2 errors as stage 1).
  - V2: adapter **rejected** (synthetic val 99.4% → 98.1%, −1.2 pts), so stage 2 = stage 1 = **100.0%** and `app.py` runs V2 without an adapter.
  - The adapters fit their own finetune files poorly (V1 67.7%, V2 62.9%); half the finetune set is now copies of one recording, so more real Very Unsatisfied / Very Satisfied calls would help more than further copies. This was the second stage-2 variant compared on the same 13 test calls, so these numbers are optimistic. The previous adapters are in `models_backup_adapters_20261017/`.
- **Benchmark Accuracies, 13-call real test (2026-10-17, first run):** 4 new uploads (`SampleInboundCall2`, `OUTBOUNDSAMPLE_01`, `OUTBOUNDSAMPLE_02`, `InboundSampleRecording`, all Satisfied) ingested, each upload augmented ×5, real calls re-split as above. Same stage-1 models (not retrained); stage 2 retrained on the 32 finetune files (`--ft_anchor_weight 1.0`).
  - V1: **99.2% synthetic val** (98.99% with adapter, accepted). Real test **84.6% stage 1 (11/13) → 76.9% stage 2 (10/13)**. Stage 1 misses `yt_dGFK8vGxX7g` (Unsatisfied → Very Satisfied) and `yt_gD7xQGXpSBg` (Very Satisfied → Unsatisfied); the adapter keeps both errors and adds `yt_DwOVXz74N9c` (Very Unsatisfied → Satisfied).
  - V2: **99.4% synthetic val** (98.8% with adapter, accepted). Real test **100.0% stage 1 (13/13) → 84.6% stage 2 (11/13)**; the adapter breaks `yt_Xi44fYgJYSU` (Satisfied → Unsatisfied) and `yt_gD7xQGXpSBg` (Very Satisfied → Unsatisfied).
  - Both adapters were kept at the time (they passed the pre-set synthetic-val guard, even though they lowered real-test accuracy); replaced by the run above. The finetune set is heavily skewed (22 of 32 Satisfied, 1 Very Unsatisfied), which likely pulls the adapter toward Satisfied/Unsatisfied; V2's adapter only reaches 66% on its own finetune files. With 13 test calls, one call = 7.7 pts.
  - Not comparable with the 15- or 19-call numbers: the test set has different calls (V1's stage-1 drop from 93.3% to 84.6% is only `yt_dGFK8vGxX7g` returning to test).
- **Benchmark Accuracies, 15-call YouTube test (2026-10-16):** same stage-1 models as below (synthetic data unchanged, so stage 1 was not retrained); stage 2 retrained on the 6 finetune calls.
  - V1: **99.2% synthetic val**, **93.3% YouTube (14/15)**. The adapter fits the 6 calls (100%) but lowers synthetic val to 97.3% (−1.9 pts), so the guard rejects it, also with `--ft_anchor_weight 3`. Stage 2 = stage 1. The only error is the angry dispute `yt_gD7xQGXpSBg` (Very Satisfied → Unsatisfied).
  - V2: **99.4% synthetic val**, **100.0% YouTube (15/15)**. Adapter rejected (98.1% val at anchor 1, 98.3% at anchor 3; the limit is a 1.0-pt drop). Stage 2 = stage 1.
  - Not comparable with the 19-call numbers: V1's rise from 89.5% to 93.3% is only because its error `yt_dGFK8vGxX7g` moved to finetune. No model changed. The previous models (with the old V1 adapter) are in `models_backup_before_yt_finetune/`.
- **Benchmark Accuracies, 19-call YouTube test (MPS GPU, two-stage recipe, `d_model=256`, 2026-09-29):** 2,569 synthetic train / 484 synthetic val, 2 uploads for stage 2, 19 YouTube test calls. Models are ~2.1M (V1) / ~2.7M (V2) parameters; the adapter is ~17k.
  - V1 Baseline (Mean Pooling): **99.2% synthetic val**, **89.5% YouTube (stage 1)**, **89.5% YouTube (stage 2)**; the adapter was accepted (synthetic val 98.8% with it) but changes no YouTube prediction.
  - V2 Upgraded (`MLPResBlock` + `[CLS]`): **99.4% synthetic val**, **100.0% YouTube (stage 1, 19/19)**; no stage-2 adapter was saved (rejected by the guard), so stage 2 = stage 1.
  - V1's 2 remaining YouTube errors: `yt_dGFK8vGxX7g` (Unsatisfied → Very Satisfied) and the angry dispute `yt_gD7xQGXpSBg` (Very Satisfied → Unsatisfied).
  - Previous run at `d_model=512` (2026-09-28): V1 97.7% val / 78.9% YouTube, V2 99.4% val / 78.9% YouTube.
  - Earlier numbers (40% / 20% YouTube, then 47.4% / 42.1%) were measured against the unswapped YouTube labels and are not comparable.

## 5. Augment & Retrain (step list)
Run from the repo root with the `.venv` active.
```bash
source .venv/bin/activate && pip install -r requirements.txt   # first time: see Step 0 in section 3
python scripts/check_gpu.py --quick                            # confirm MPS works before a long run
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
# 6. Check in the app
streamlit run app.py
```
