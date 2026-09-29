import os
import glob
import json
import random
import shutil
import hashlib
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from paths import SPLIT_MANIFEST_PATH, DIALOGUES_PATH, FEATURES_DIR
REAL_WORLD_PREFIXES = ("yt_", "upload_")
FINETUNE_PREFIX = "upload_"  # real uploaded calls: train the stage-2 adapter
TEST_PREFIX = "yt_"          # real YouTube calls: final test set, never trained on
SPLITS = ("train", "val", "finetune", "test")


def _source_id(basename):
    """'dial_001_x__aug03.pt' -> 'dial_001_x'; augmented variants share their source's id."""
    return os.path.splitext(basename)[0].split("__aug")[0]


def _script_groups():
    """Maps dialogue_id -> hash of its turn texts, so re-voiced copies of one script share a group."""
    groups = {}
    if os.path.exists(DIALOGUES_PATH):
        with open(DIALOGUES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                joined = "||".join(t["text"].strip().lower() for t in d["turns"])
                groups[d["dialogue_id"]] = hashlib.sha1(joined.encode("utf-8")).hexdigest()
    return groups


def get_or_create_splits(features_dir=FEATURES_DIR, val_ratio=0.15, seed=42, rebuild=False):
    """
    Creates or loads a deterministic split manifest for the two-stage training recipe.
    - train/val: synthetic TTS dialogues (dial*) and their augmented variants (stage 1: pretraining).
      val_ratio of the unique *scripts* per class are held out, and every re-voiced copy
      (dial_aug_*) and augmented variant (__augNN) of a script stays on the same side,
      so no dialogue text is shared between train and val.
    - finetune: every uploaded real call (upload_*) (stage 2: trains the adapter on the frozen model).
    - test: every YouTube call (yt_*) (stage 3: final evaluation, never trained on).
    An older manifest without all four splits is backed up and regenerated automatically.
    """
    if os.path.exists(SPLIT_MANIFEST_PATH):
        try:
            with open(SPLIT_MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if all(k in manifest for k in SPLITS):
                if not rebuild:
                    return manifest
            else:
                version = "v2" if "val" in manifest else "v1"
                backup = SPLIT_MANIFEST_PATH.replace(".json", f".{version}.json")
                if not os.path.exists(backup):
                    shutil.copy(SPLIT_MANIFEST_PATH, backup)
                print(f"ℹ️ Older split manifest found; backed up to {backup} and regenerating train/val/finetune/test.")
        except Exception:
            pass
            
    all_files = sorted(glob.glob(os.path.join(features_dir, "*.pt")))
    if not all_files:
        return {k: [] for k in SPLITS}
        
    random.seed(seed)
    script_groups = _script_groups()
    
    finetune_files = []
    test_files = []
    # label -> group key -> files
    syn_groups = defaultdict(lambda: defaultdict(list))
    
    for f in all_files:
        basename = os.path.basename(f)
        if basename.startswith(FINETUNE_PREFIX):
            finetune_files.append(f)
            continue
        if basename.startswith(TEST_PREFIX):
            test_files.append(f)
            continue
        data = torch.load(f, map_location="cpu")
        label = data["label"].item() if isinstance(data["label"], torch.Tensor) else int(data["label"])
        source_id = data.get("source_id", _source_id(basename))
        group = script_groups.get(source_id, source_id)
        syn_groups[label][group].append(f)
            
    train_files = []
    val_files = []
    
    # Hold out val_ratio of the unique scripts per class (at least 1)
    for label in sorted(syn_groups):
        groups = sorted(syn_groups[label])
        random.shuffle(groups)
        n_val = max(1, round(len(groups) * val_ratio)) if len(groups) > 1 else 0
        for i, group in enumerate(groups):
            (val_files if i < n_val else train_files).extend(syn_groups[label][group])
        
    split_manifest = {
        "train": sorted(train_files),
        "val": sorted(val_files),
        "finetune": sorted(finetune_files),
        "test": sorted(test_files)
    }
    
    os.makedirs(os.path.dirname(SPLIT_MANIFEST_PATH), exist_ok=True)
    with open(SPLIT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)
        
    return split_manifest


class RealCSATDataset(Dataset):
    """
    Loads extracted multimodal PyTorch tensors with strict conversation-level isolation.
    """
    def __init__(self, features_dir=FEATURES_DIR, split="all", val_ratio=0.15, seed=42):
        self.split = split
        
        if split in SPLITS:
            splits = get_or_create_splits(features_dir=features_dir, val_ratio=val_ratio, seed=seed)
            self.files = splits[split]
        elif split == "all":
            self.files = sorted(glob.glob(os.path.join(features_dir, "*.pt")))
        else:
            raise ValueError(f"Unknown split: '{split}'. Must be one of {SPLITS} or 'all'.")
            
        if len(self.files) == 0:
            raise RuntimeError(f"No feature files found for split='{split}' in {features_dir}!")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")
        
        # Dual Transformer format
        audio_embeds = data.get("audio_embeds", None)
        text_embeds = data.get("text_embeds", None)
        
        # Legacy Hybrid CNN format fallback
        if audio_embeds is None and "mel_spec" in data:
            audio_embeds = data.get("mel_spec")
        if text_embeds is None and "chunked_text" in data:
            text_embeds = data.get("chunked_text")
            
        label = data["label"]
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        return text_embeds, audio_embeds, label


def pad_collate(batch):
    """
    Custom collate_fn to pad varying sequence lengths across dialogues and generate padding masks.
    """
    text_embeds_list = [item[0] for item in batch]
    audio_embeds_list = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    batch_size = len(batch)
    lengths = [t.size(0) for t in text_embeds_list]
    max_len = max(lengths)
    
    # Pad sequences
    text_padded = pad_sequence(text_embeds_list, batch_first=True, padding_value=0.0)
    audio_padded = pad_sequence(audio_embeds_list, batch_first=True, padding_value=0.0)
    
    # Construct boolean padding mask (True where index >= sequence length)
    padding_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, l in enumerate(lengths):
        if l < max_len:
            padding_mask[i, l:] = True
            
    labels_stacked = torch.stack(labels)
    
    return text_padded, audio_padded, padding_mask, labels_stacked


