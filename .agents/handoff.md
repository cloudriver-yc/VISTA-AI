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
├── app.py                      # Interactive Streamlit Web UI (Side-by-side V1 vs V2)
├── data/                       # Datasets & Metadata
│   ├── audio/out/              # Synthesized WAV files (stereo)
│   ├── audio/youtube/          # Downloaded YouTube 16kHz audio & captions/
│   ├── features/               # 326 extracted PyTorch tensors (audio_embeds, text_embeds, label)
│   ├── raw/                    # Raw JSONL transcripts and metadata
│   ├── train_test_split.json   # Zero-leakage conversation-level train/test split manifest
│   └── youtube_metadata.jsonl  # YouTube titles, captions, and Whisper transcripts
├── models/                     # Trained checkpoints (dual_transformer_v1_weights.pt, v2_weights.pt)
├── scripts/                    # Pipeline Scripts
│   ├── 01_generate_scripts.py     # Generates initial CSAT dialogues via Gemini
│   ├── 02_augment_scripts.py      # Expands synthetic dataset to 500+ samples
│   ├── 03_synthesize_audio.py     # Converts JSON transcripts to TTS audio via ElevenLabs
│   ├── 04_extract_features.py     # Extracts WavLM & Text embeddings from audio/JSON
│   ├── 05_test_youtube.py         # Side-by-side V1 vs V2 validation script with telemetry
│   ├── evaluate_asr.py            # Evaluates Whisper ASR accuracy vs YouTube captions (WER)
│   ├── ingest_youtube_dataset.py  # Ingests YouTube playlists, removes B-roll & extracts features
│   └── hf_dataset_sync.py         # Hugging Face upload/download utility
└── src/                        # Core PyTorch Framework
    ├── data/
    │   └── dataset.py          # Custom PyTorch Dataset (RealCSATDataset) with conversation isolation
    ├── models/
    │   └── architecture.py     # DualTransformerClassifier (V1), EnhancedDualTransformerClassifier (V2), MLPResBlock
    └── train.py                # Comparative training loop (Run via `python src/train.py --model_version both`)
```

## 3. The Execution Workflow

### Step 1: Download or Ingest Data
- **Hugging Face Sync:**
  ```bash
  python scripts/hf_dataset_sync.py download --repo-id Vista-AI/CustomerServiceAudio
  ```
- **Ingest Real YouTube Playlists:**
  ```bash
  python scripts/ingest_youtube_dataset.py
  ```

### Step 2: Feature Extraction (Whisper ASR + WavLM + SentenceTransformer)
Extract multimodal segment representations:
```bash
python scripts/04_extract_features.py
```

### Step 3: Train Both V1 Baseline and V2 Upgraded Models
Train both architectures side-by-side on Apple Silicon GPU (`mps`):
```bash
python src/train.py --model_version both --epochs 25 --batch_size 16 --lr 2e-4
```
Weights will be saved to `models/dual_transformer_v1_weights.pt` and `models/dual_transformer_v2_weights.pt`.

### Step 4: Side-by-Side Inference & Telemetry
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
- **Dataset Size:** 327 multimodal dialogue samples (307 synthetic in `data/audio/out/` + 19 real-world YouTube dialogues in `data/audio/youtube/` + 1 uploaded real-world call in `data/features/upload_1755884171_51632.pt`).
- **Train/Test Isolation (`data/train_test_split.json`):**
  - **Train Set:** 279 dialogues (264 synthetic + 14 YouTube + 1 uploaded real call).
  - **Test Set:** 48 dialogues (43 synthetic + 5 held-out YouTube).
  - **YouTube Preservation:** Every category has at least 1 full video held out exclusively for test.
  - **Zero Leakage:** 0 files shared between train and test.
- **Benchmark Accuracies (MPS GPU):**
  - V1 Baseline (Mean Pooling): **100.0% Train Acc**, **91.7% Test Acc** (`models/dual_transformer_v1_weights.pt`).
  - V2 Upgraded (`MLPResBlock` + `[CLS]`): **100.0% Train Acc**, **91.7% Test Acc** (`models/dual_transformer_v2_weights.pt`).
- **Real-World Dispute Test:** Both V1 (99.65%) and V2 (99.31%) correctly predict `Very Unsatisfied`.







