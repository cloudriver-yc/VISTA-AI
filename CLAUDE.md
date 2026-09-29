# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

VISTA-AI predicts a 4-class Customer Satisfaction (CSAT) label from conversational call audio by fusing per-turn text semantics and acoustic prosody. Two PyTorch architectures (V1 and V2) are trained and compared side by side everywhere: in training, in the CLI test and in the Streamlit UI.

Deeper context lives in `.agents/`: `handoff.md` has the architecture, dataset counts and benchmark numbers, `walkthrough.md` covers the XAI layer, and `xai_implementation_plan.md` has the XAI design.

## Workspace rule (from `.agents/AGENTS.md`)

After making any code or structural change, **always ask the user** whether the docs in `.agents/` (`handoff.md`, `walkthrough.md`, `implementation_plan.md`, etc.) should be updated to match. Recent commits show these docs are kept in sync with dataset and model changes, including dataset sizes, split counts and accuracies.

## Commands

Run all commands from the repo root. Paths such as `data/features` and `models/` are relative to the root. Use the `.venv`. `ffmpeg` must be on PATH, and API keys (`HF_TOKEN`, `ELEVENLABS_API_KEY`, optional `ANTHROPIC_API_KEY`) are read from `.env`. Script generation uses the Antigravity CLI (`agy`), logged in with the user's Google account; no API key is needed.

```bash
pip install -r requirements.txt

# Data: audio, caches, features and models/ are gitignored and synced via Hugging Face. All paths live in src/paths.py.
python scripts/hf_dataset_sync.py download [--parts metadata features models]   # parts: metadata features models synthetic-audio real-audio (default all); auto-runs the migration
python scripts/hf_dataset_sync.py upload --dry-run          # then without --dry-run; features/ and models/ are mirrored, legacy-layout copies pruned
python scripts/hf_dataset_sync.py status --remote           # local vs Hub per part
python scripts/migrate_data_layout.py [--dry-run]   # legacy data/audio, data/raw layout -> data/synthetic + data/real
python scripts/ingest_youtube_dataset.py          # real YouTube calls -> data/real/youtube + features
python scripts/01_generate_scripts.py --per-class 60 --long-frac 0.4   # new unique scripts via Antigravity CLI; re-run to resume after a quota stop
python scripts/03_synthesize_audio.py --only-missing   # voice new/unvoiced unique scripts (ElevenLabs)
python scripts/03b_augment_audio.py --download-noise --target 2500   # acoustic augmentation -> data/synthetic/augmented
python scripts/04_extract_features.py --only syn,aug --skip-existing # Whisper ASR -> WavLM + MPNet per-segment features

# Two-stage training (v1 | v2 | both): stage 1 -> models/dual_transformer_{v1,v2}_weights.pt, stage 2 -> models/csat_adapter_{v1,v2}.pt
python src/train.py --stage all --rebuild_split            # or --stage pretrain | finetune
python scripts/ingest_upload.py --audio call.mp3 --label unsatisfied   # add a real uploaded call to the stage-2 pool

# Evaluate / run
python scripts/05_test_youtube.py                 # CLI V1-vs-V2 comparison with telemetry
python scripts/evaluate_asr.py                    # Whisper WER vs YouTube captions
streamlit run app.py                              # dashboard: upload / mic / YouTube URL
```

The repo has no test suite, linter or build step. Checking a change means retraining and reading the train, synthetic-val and real-world test accuracies that `src/train.py` prints, or running the app.

`scripts/01`–`03b` form the synthetic-data pipeline: `01_generate_scripts.py` writes new unique scripts through the Antigravity CLI (`agy -p --json-schema`, models in priority order `gemini-3.8-flash-medium` > `gemini-3.8-flash-high` > `gemini-3.1-pro-high` > `claude-sonnet-4-6`, falling back only when a quota runs out (`--rotate` spreads requests round-robin); a worker pool keeps `--concurrency` requests in flight; `--provider anthropic-api` uses the Anthropic SDK instead). Output is validated against the `Dialogue` pydantic schema, and `--per-class` is a resumable per-class target. `agy` runs headless from the empty `data/tmp/agy_workspace/` with skills disabled and a no-tools instruction, because otherwise the agent tries to read files or run commands, which headless mode denies. ElevenLabs then voices them as stereo TTS (`03`), and `03b_augment_audio.py` makes class-balanced mono variants (mixdown, speed/pitch, reverb, ESC-50/colored/hum/babble noise, telephone band, gain) with the exact recipe per clip in `data/synthetic/augmented/manifest.jsonl`. The `dial_aug_*` rows in `dialogues.jsonl` came from a since-removed `02_augment_scripts.py` that re-voiced existing scripts with identical text, which is why the original 307 TTS clips cover just 16 unique scripts. The split groups them with their base script.

## Architecture

**Pipeline:** the audio is converted to 16 kHz mono (stereo TTS is summed across both speakers by default; `--channel customer` keeps ch0 only) and transcribed by Whisper (`small.en` in the app). For each Whisper segment, the code computes a 768-d WavLM (`microsoft/wavlm-base-plus`) mean embedding of that audio slice and a 768-d MPNet (`all-mpnet-base-v2`) embedding of the segment text. It then stacks them into `(num_segments, 768)` tensors and saves `{"audio_embeds", "text_embeds", "label"}` as one `.pt` file per conversation in `data/features/`. One file is one dialogue, and the sequence dimension is dialogue turns, not time frames.

**Models (`src/models/architecture.py`):**
- `DualTransformerClassifier` (V1): linear projections to `d_model=512`, then bidirectional `CrossModalAttention` (text↔audio), sinusoidal positional encoding, a 2-layer Transformer encoder over turns, mean pooling, and a 4-class head.
- `EnhancedDualTransformerClassifier` (V2): the same design, with an added `MLPResBlock` adaptation head and a learnable `[CLS]` token that is used for pooling.
- `forward(text, audio, padding_mask=None, return_xai=False)` returns `(logits, None, None)` by default, and training and the scripts unpack three values. With `return_xai=True`, it returns a dict with turn saliency, text/audio norms, modality ratio and per-turn cross-modal mismatch. `src/models/explainer.py` (`MultimodalExplainer`) consumes that dict to produce the pivotal turns, tone diagnoses and narrative shown in the app. Keep the default return signature backward compatible.
- `src/train.py` passes explicit hyperparameters (`d_model=512, nhead=8, num_layers=2, dropout=0.3`). `app.py` and `scripts/05_test_youtube.py` rely on the constructor defaults instead. If you change the architecture shape, keep the defaults and the train args in sync, or the saved state_dicts won't load.

**Label mapping is intentionally non-intuitive.** This mapping is canonical in `src/feature_extraction.py` `LABEL_MAP` and `explainer.CANONICAL_CLASSES`. Don't "fix" it:
`0 "Very Unsatisfied"` = happy/delighted, resolved; `1 "Unsatisfied"` = flat, unresolved; `2 "Satisfied"` = flat, resolved; `3 "Very Satisfied"` = angry/shouting, unresolved. Legacy aliases (`promoter_delighted`, `urgent_follow_up`, …) map onto the same indices.

**Train/val/finetune/test split (`src/data/dataset.py`):** `get_or_create_splits` writes `{"train","val","finetune","test"}` to `data/split_manifest.json`. **finetune** = uploaded real calls (`upload_*`), used only by stage 2. **test** = YouTube calls (`yt_*`), never trained on. **train/val** = synthetic (`dial*`) only, with ~15% of the unique *scripts* per class held out as val. Files are grouped by a hash of the script's turn text from `dialogues.jsonl`, so `dial_aug_*` re-voicings and `__augNN` augmented variants always sit on the same side as their base script. **Once the manifest exists it is loaded as-is**, so new feature files are ignored until you run `src/train.py --rebuild_split` (or delete the manifest). An older manifest missing any of the four splits is backed up (`split_manifest.v1.json` / `.v2.json`) and regenerated automatically. The `split="all"` option globs the directory directly. Feature filename prefixes matter: `dial*` = synthetic (including `{source_id}__augNN.pt`), `yt_*` = YouTube, `upload_*` = real uploaded calls. The YouTube playlists were curated with the everyday meaning of the extreme classes, so their labels are swapped 0 ↔ 3 to the project mapping (in `ingest_youtube_dataset.py` and in the existing `yt_*.pt`, marked `label_convention: "project"`). `pad_collate` pads variable turn counts and builds a `True`-means-padding mask.

**Training (`src/train.py`, two stages):** stage 1 (`--stage pretrain`) trains V1/V2 on synthetic data and keeps the best synthetic-val checkpoint. Stage 2 (`--stage finetune`) wraps the frozen stage-1 model in `AdaptedCSATModel` with a `ResidualAdapterHead` on the 512-d summary from the new `encode()` method (V1 pooled / V2 `[CLS]`). The adapter output is added to the frozen logits, its last layer is zero-initialised (so an untrained adapter changes nothing), and only the adapter trains on the finetune split, with an anchor KL penalty toward the frozen model's predictions on synthetic training calls (`--ft_anchor_weight`) and a guard that rejects the adapter if synthetic-val accuracy drops by more than `--ft_max_val_drop` points (stage 2 then falls back to stage 1). Never tune stage-2 settings on YouTube; it is the test set. Every run ends by scoring stage 1 and stage 2 per file on the YouTube test set. `app.py` and `05_test_youtube.py` wrap models with `with_adapter_if_available()`. Feature extraction lives in `src/feature_extraction.py` (shared by `04_extract_features.py` and `ingest_upload.py`).

**Data layout (`src/paths.py`):** `data/synthetic/` is the training pool (`dialogues.jsonl`, `tts/`, `tts_turn_cache/`, `augmented/`); `data/real/` is the test pool (`youtube/{audio,captions,metadata.jsonl}`, `uploads/{audio,metadata.jsonl}`); plus `data/features/`, `data/noise/esc50/`, `data/split_manifest.json` and `data/tmp/` (app scratch). Use the constants in `paths.py` instead of hard-coding paths. The `app_rk*.py` / `app07092026 - bk_.py` backups still reference the legacy layout.

**Weights:** `src/train.py` saves V1/V2 weights and also copies V2 to `models/dual_transformer_weights.pt` and `models/hybrid_cnn_weights.pt`, which are legacy names still used as fallbacks. Device selection is always MPS → CUDA → CPU.

**`app.py` (Streamlit):** caches all models via `@st.cache_resource` and adds `src/` to `sys.path` for imports. It resolves input with the priority upload > mic recording > YouTube URL and normalizes it with ffmpeg. It runs Whisper on a background thread with a live elapsed-time display, runs V1 and V2 one after the other with `return_xai=True`, and assigns heuristic speaker labels by KMeans-clustering the per-segment WavLM embeddings, choosing k by silhouette score. There is no real diarization model. `app_rk.py`, `app_rk_07092026.py` and `app07092026 - bk_.py` are old backup copies; edit `app.py`. For Streamlit work, the `developing-with-streamlit` skill in `.claude/skills/` applies.
