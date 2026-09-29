"""
Single source of truth for data and model paths (relative to the repo root).

data/
├── synthetic/                  # Training pool: generated scripts -> TTS -> augmentation
│   ├── dialogues.jsonl         #   Gemini dialogue scripts (labels, turns, voices)
│   ├── tts/                    #   ElevenLabs stereo WAVs (customer ch0, engineer ch1)
│   ├── tts_turn_cache/         #   Per-turn TTS cache (*_turn_N.npy), reused for babble noise
│   ├── augmented/              #   Augmented mono FLAC clips + manifest.jsonl
│   └── augmented_preview/      #   Small WAV batch for listening QA
├── real/                       # Test pool: real-world calls, never trained on
│   ├── youtube/{audio,captions}/ + metadata.jsonl
│   └── uploads/audio/ + metadata.jsonl
├── noise/esc50/                # Background-noise bank for augmentation
├── features/                   # One .pt per dialogue: dial* (synthetic), yt_* / upload_* (real)
├── split_manifest.json         # train / val / test file lists
└── tmp/                        # Scratch for app.py and CLI downloads (safe to delete)
"""

DATA_DIR = "data"

# Synthetic training pool
SYNTHETIC_DIR = f"{DATA_DIR}/synthetic"
DIALOGUES_PATH = f"{SYNTHETIC_DIR}/dialogues.jsonl"
TTS_DIR = f"{SYNTHETIC_DIR}/tts"
TTS_TURN_CACHE_DIR = f"{SYNTHETIC_DIR}/tts_turn_cache"
AUGMENTED_DIR = f"{SYNTHETIC_DIR}/augmented"
AUGMENTED_MANIFEST_PATH = f"{AUGMENTED_DIR}/manifest.jsonl"
AUGMENTED_PREVIEW_DIR = f"{SYNTHETIC_DIR}/augmented_preview"

# Real-world test pool
REAL_DIR = f"{DATA_DIR}/real"
YOUTUBE_DIR = f"{REAL_DIR}/youtube"
YOUTUBE_AUDIO_DIR = f"{YOUTUBE_DIR}/audio"
YOUTUBE_CAPTIONS_DIR = f"{YOUTUBE_DIR}/captions"
YOUTUBE_METADATA_PATH = f"{YOUTUBE_DIR}/metadata.jsonl"
UPLOADS_DIR = f"{REAL_DIR}/uploads"
UPLOADS_AUDIO_DIR = f"{UPLOADS_DIR}/audio"
UPLOADS_METADATA_PATH = f"{UPLOADS_DIR}/metadata.jsonl"

# Shared
NOISE_DIR = f"{DATA_DIR}/noise/esc50"
FEATURES_DIR = f"{DATA_DIR}/features"
SPLIT_MANIFEST_PATH = f"{DATA_DIR}/split_manifest.json"
TMP_DIR = f"{DATA_DIR}/tmp"

# Models
MODELS_DIR = "models"
