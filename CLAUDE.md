# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

VISTA-AI predicts a 4-class Customer Satisfaction (CSAT) label from conversational call audio by fusing per-turn text semantics and acoustic prosody. Two PyTorch architectures (V1 and V2) are trained and compared side by side everywhere: in training, in the CLI test and in the Streamlit UI.

Deeper context lives in `.agents/`: `handoff.md` has the architecture, dataset counts and benchmark numbers, `walkthrough.md` covers the XAI layer, and `xai_implementation_plan.md` has the XAI design.

## Workspace rule (from `.agents/AGENTS.md`)

After making any code or structural change, **always ask the user** whether the docs in `.agents/` (`handoff.md`, `walkthrough.md`, `implementation_plan.md`, etc.) should be updated to match. Recent commits show these docs are kept in sync with dataset and model changes, including dataset sizes, split counts and accuracies.

## Commands

Run all commands from the repo root. Paths such as `data/features` and `models/` are relative to the root. Use the `.venv`. `ffmpeg` must be on PATH, and API keys (`HF_TOKEN`, `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`) are read from `.env`.

```bash
pip install -r requirements.txt

# Data (data/audio, data/features and models/ are gitignored; they're synced via Hugging Face)
python scripts/hf_dataset_sync.py download --repo-id Vista-AI/CustomerServiceAudio   # also: upload | status
python scripts/ingest_youtube_dataset.py          # real YouTube calls -> data/audio/youtube + features
python scripts/04_extract_features.py             # Whisper ASR -> WavLM + MPNet per-segment features

# Train (v1 | v2 | both); writes models/dual_transformer_{v1,v2}_weights.pt
python src/train.py --model_version both --epochs 25 --batch_size 16 --lr 2e-4

# Evaluate / run
python scripts/05_test_youtube.py                 # CLI V1-vs-V2 comparison with telemetry
python scripts/evaluate_asr.py                    # Whisper WER vs YouTube captions
streamlit run app.py                              # dashboard: upload / mic / YouTube URL
```

The repo has no test suite, linter or build step. Checking a change means retraining and reading the train, test and held-out YouTube accuracies that `src/train.py` prints, or running the app.

`scripts/01`–`03` form the synthetic-data generation pipeline: Gemini generates dialogue scripts, they are augmented, and ElevenLabs turns them into TTS audio. It only needs re-running to grow the synthetic set.

## Architecture

**Pipeline:** the audio is converted to 16 kHz mono and transcribed by Whisper (`small.en` in the app). For each Whisper segment, the code computes a 768-d WavLM (`microsoft/wavlm-base-plus`) mean embedding of that audio slice and a 768-d MPNet (`all-mpnet-base-v2`) embedding of the segment text. It then stacks them into `(num_segments, 768)` tensors and saves `{"audio_embeds", "text_embeds", "label"}` as one `.pt` file per conversation in `data/features/`. One file is one dialogue, and the sequence dimension is dialogue turns, not time frames.

**Models (`src/models/architecture.py`):**
- `DualTransformerClassifier` (V1): linear projections to `d_model=512`, then bidirectional `CrossModalAttention` (text↔audio), sinusoidal positional encoding, a 2-layer Transformer encoder over turns, mean pooling, and a 4-class head.
- `EnhancedDualTransformerClassifier` (V2): the same design, with an added `MLPResBlock` adaptation head and a learnable `[CLS]` token that is used for pooling.
- `forward(text, audio, padding_mask=None, return_xai=False)` returns `(logits, None, None)` by default, and training and the scripts unpack three values. With `return_xai=True`, it returns a dict with turn saliency, text/audio norms, modality ratio and per-turn cross-modal mismatch. `src/models/explainer.py` (`MultimodalExplainer`) consumes that dict to produce the pivotal turns, tone diagnoses and narrative shown in the app. Keep the default return signature backward compatible.
- `src/train.py` passes explicit hyperparameters (`d_model=512, nhead=8, num_layers=2, dropout=0.3`). `app.py` and `scripts/05_test_youtube.py` rely on the constructor defaults instead. If you change the architecture shape, keep the defaults and the train args in sync, or the saved state_dicts won't load.

**Label mapping is intentionally non-intuitive.** This mapping is canonical in `scripts/04_extract_features.py` `LABEL_MAP` and `explainer.CANONICAL_CLASSES`. Don't "fix" it:
`0 "Very Unsatisfied"` = happy/delighted, resolved; `1 "Unsatisfied"` = flat, unresolved; `2 "Satisfied"` = flat, resolved; `3 "Very Satisfied"` = angry/shouting, unresolved. Legacy aliases (`promoter_delighted`, `urgent_follow_up`, …) map onto the same indices.

**Train/test split (`src/data/dataset.py`):** `get_or_create_splits` builds a conversation-level split, stratified per class, with at least one YouTube video per class held out for test. It is persisted to `data/train_test_split.json`. **Once that manifest exists it is loaded as-is**, so a new feature file stays out of train/test until you add it to the manifest by hand, which is how uploaded calls were integrated. The alternative is to delete the manifest and regenerate it, which reshuffles the whole split. The `split="all"` option globs the directory directly. Feature filename prefixes matter: `dial*` = synthetic, `yt_*` = YouTube (evaluated separately as "held-out YouTube accuracy"), `upload_*` = real uploaded calls. `pad_collate` pads variable turn counts and builds a `True`-means-padding mask.

**Weights:** `src/train.py` saves V1/V2 weights and also copies V2 to `models/dual_transformer_weights.pt` and `models/hybrid_cnn_weights.pt`, which are legacy names still used as fallbacks. Device selection is always MPS → CUDA → CPU.

**`app.py` (Streamlit):** caches all models via `@st.cache_resource` and adds `src/` to `sys.path` for imports. It resolves input with the priority upload > mic recording > YouTube URL and normalizes it with ffmpeg. It runs Whisper on a background thread with a live elapsed-time display, runs V1 and V2 one after the other with `return_xai=True`, and assigns heuristic speaker labels by KMeans-clustering the per-segment WavLM embeddings, choosing k by silhouette score. There is no real diarization model. `app_rk.py`, `app_rk_07092026.py` and `app07092026 - bk_.py` are old backup copies; edit `app.py`. For Streamlit work, the `developing-with-streamlit` skill in `.claude/skills/` applies.
