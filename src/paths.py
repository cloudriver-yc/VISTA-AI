"""
Single source of truth for data and model paths (relative to the repo root).

data/
├── synthetic/                  # Training pool: generated scripts -> TTS -> augmentation
│   ├── dialogues.jsonl         #   Gemini dialogue scripts (labels, turns, voices)
│   ├── tts/                    #   ElevenLabs stereo WAVs (customer ch0, engineer ch1)
│   ├── tts_turn_cache/         #   Per-turn TTS cache (*_turn_N.npy), reused for babble noise
│   ├── augmented/              #   Augmented mono FLAC clips + manifest.jsonl
│   └── augmented_preview/      #   Small WAV batch for listening QA
├── real/                       # Real-world calls
│   ├── youtube/{audio,captions}/ + metadata.jsonl   #   YouTube calls (finetune or test)
│   │   └── augmented/          #   {video_id}__augNN.flac + manifest.jsonl (03b_augment_audio.py --youtube ID)
│   └── uploads/audio/ + metadata.jsonl              #   manual uploads
│       └── augmented/          #   {upload_id}__augNN.flac + manifest.jsonl (03b_augment_audio.py --uploads N)
├── noise/esc50/                # Background-noise bank for augmentation
├── features/                   # One .pt per dialogue, in the folder of the split that uses it:
│   ├── train/                  #   dial*    synthetic, stage-1 training
│   ├── val/                    #   dial*    synthetic, held-out scripts (model selection)
│   ├── finetune/               #   real calls (upload_* / yt_*) + their augmented copies, stage-2 adapter
│   ├── test/                   #   real calls, final test, never trained on (originals only)
│   ├── unassigned/             #   new files not used yet; `src/train.py --rebuild_split` sorts them
│   └── excluded/               #   augmented copies of test calls: never used, they would leak the test set
├── split_manifest.json         # Record of the folders above (rewritten whenever they change)
└── tmp/                        # Scratch for app.py and CLI downloads (safe to delete)
"""
import os

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
YOUTUBE_AUGMENTED_DIR = f"{YOUTUBE_DIR}/augmented"
YOUTUBE_AUGMENTED_MANIFEST_PATH = f"{YOUTUBE_AUGMENTED_DIR}/manifest.jsonl"
UPLOADS_DIR = f"{REAL_DIR}/uploads"
UPLOADS_AUDIO_DIR = f"{UPLOADS_DIR}/audio"
UPLOADS_METADATA_PATH = f"{UPLOADS_DIR}/metadata.jsonl"
UPLOADS_AUGMENTED_DIR = f"{UPLOADS_DIR}/augmented"
UPLOADS_AUGMENTED_MANIFEST_PATH = f"{UPLOADS_AUGMENTED_DIR}/manifest.jsonl"

# Shared
NOISE_DIR = f"{DATA_DIR}/noise/esc50"
FEATURES_DIR = f"{DATA_DIR}/features"
FEATURE_SPLITS = ("train", "val", "finetune", "test")
FEATURES_UNASSIGNED_DIR = f"{FEATURES_DIR}/unassigned"
FEATURES_EXCLUDED_DIR = f"{FEATURES_DIR}/excluded"
FINETUNE_PREFIX = "upload_"  # manual uploads always go to finetune/
TEST_PREFIX = "yt_"          # YouTube calls always go to test/
SPLIT_MANIFEST_PATH = f"{DATA_DIR}/split_manifest.json"
TMP_DIR = f"{DATA_DIR}/tmp"

# Models
MODELS_DIR = "models"


def feature_split_dir(split):
    """data/features/<split> for one of FEATURE_SPLITS."""
    return f"{FEATURES_DIR}/{split}"


def feature_dirs():
    """Every folder a feature file can live in: the four splits, unassigned/ and excluded/."""
    return [feature_split_dir(s) for s in FEATURE_SPLITS] + [FEATURES_UNASSIGNED_DIR, FEATURES_EXCLUDED_DIR]


def feature_source_id(name):
    """'upload_x__aug03.pt' -> 'upload_x': augmented copies share their original's id."""
    return os.path.splitext(name)[0].split("__aug")[0]


def is_augmented(name):
    return "__aug" in name


def find_feature(name):
    """Current path of feature file `name` (e.g. 'dial_001_x.pt'), or None if it hasn't been extracted."""
    for d in feature_dirs():
        if os.path.exists(f"{d}/{name}"):
            return f"{d}/{name}"
    return None


def feature_home(name):
    """Folder a newly extracted feature file belongs in. Synthetic files wait in unassigned/ until a split rebuild.
    An augmented copy of a real call follows its original: finetune/ if the original is there, excluded/ if it is
    a test call (a copy in finetune would leak the test call into stage 2), unassigned/ if not placed yet."""
    if name.startswith((FINETUNE_PREFIX, TEST_PREFIX)) and is_augmented(name):
        original = find_feature(feature_source_id(name) + ".pt")
        if original is None:
            return FEATURES_UNASSIGNED_DIR
        folder = os.path.basename(os.path.dirname(original))
        return FEATURES_EXCLUDED_DIR if folder == "test" else feature_split_dir("finetune") if folder == "finetune" else FEATURES_UNASSIGNED_DIR
    if name.startswith(FINETUNE_PREFIX):
        return feature_split_dir("finetune")
    if name.startswith(TEST_PREFIX):
        return feature_split_dir("test")
    return FEATURES_UNASSIGNED_DIR


def feature_output_path(name):
    """Where to write feature file `name`: over its existing copy if there is one, else in its home folder."""
    return find_feature(name) or f"{feature_home(name)}/{name}"
