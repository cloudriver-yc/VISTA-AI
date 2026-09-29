import os
import re
import sys
import json
import glob
import torch
import soundfile as sf
import numpy as np
import whisper
import subprocess
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WavLMModel
from sentence_transformers import SentenceTransformer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths
from audio_utils import segment_audio

YTDLP_BIN = os.path.abspath(".venv/bin/yt-dlp")
AUDIO_OUT_DIR = paths.YOUTUBE_AUDIO_DIR
CAPTIONS_DIR = paths.YOUTUBE_CAPTIONS_DIR
FEATURES_DIR = paths.FEATURES_DIR
METADATA_FILE = paths.YOUTUBE_METADATA_PATH

# The playlists were curated with the everyday meaning of their names ("very_unsatisfied" = angry
# customers), but the project's canonical mapping is reversed for the two extreme classes
# (0 = delighted & resolved, 3 = angry & unresolved; see CLAUDE.md). The labels below are therefore
# swapped for those two playlists, so their calls get the project's labels.
PLAYLISTS = {
    "very_unsatisfied": {
        "url": "https://www.youtube.com/playlist?list=PLUo8PXyzoa18",
        "label": 3
    },
    "unsatisfied": {
        "url": "https://www.youtube.com/playlist?list=PLVeJX74nMy1w",
        "label": 1
    },
    "satisfied": {
        "url": "https://www.youtube.com/playlist?list=PLTDhXs3h7fzM",
        "label": 2
    },
    "very_satisfied": {
        "url": "https://www.youtube.com/playlist?list=PLCMpWCMAmSPc",
        "label": 0
    }
}

BROLL_PATTERNS = [
    r"\bwelcome (to|back)\b",
    r"\bsubscribe\b",
    r"\blike and subscribe\b",
    r"\bhit the (bell|like)\b",
    r"\bchannel\b",
    r"\bin this video\b",
    r"\btoday('s)? video\b",
    r"\bthanks for watching\b",
    r"\bcomment down below\b",
    r"\bsee you in the next\b",
    r"\bmusic\b",
    r"\[music\]",
    r"\[applause\]",
    r"\[laughter\]",
]

def is_broll_or_intro_outro(text, is_first_segment=False, is_last_segment=False):
    cleaned = text.strip().lower()
    if not cleaned or len(cleaned) < 3:
        return True
    for pat in BROLL_PATTERNS:
        if re.search(pat, cleaned):
            if is_first_segment or is_last_segment or "subscribe" in cleaned or "channel" in cleaned:
                return True
    return False

def download_video(video_id, url):
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    os.makedirs(CAPTIONS_DIR, exist_ok=True)
    
    audio_path = os.path.join(AUDIO_OUT_DIR, f"{video_id}.wav")
    caption_prefix = os.path.join(CAPTIONS_DIR, f"{video_id}")
    
    cmd = [
        YTDLP_BIN,
        "-x", "--audio-format", "wav",
        "--postprocessor-args", "-ar 16000 -ac 1",
        "--write-sub", "--write-auto-sub",
        "--sub-lang", "en,en-US,en-GB",
        "--sub-format", "vtt",
        "-o", audio_path.replace(".wav", ""),
        "-o", f"subtitle:{caption_prefix}",
        url
    ]
    subprocess.run(cmd, check=True, capture_output=True)
        
    caption_files = glob.glob(f"{caption_prefix}*.vtt")
    caption_text = ""
    if caption_files:
        try:
            with open(caption_files[0], "r", encoding="utf-8") as f:
                lines = f.readlines()
                filtered = [
                    re.sub(r"<[^>]+>", "", l).strip()
                    for l in lines
                    if "-->" not in l and not l.strip().isdigit() and "WEBVTT" not in l and l.strip()
                ]
                caption_text = " ".join(dict.fromkeys(filtered))
        except Exception:
            pass
            
    return audio_path, caption_text

def get_playlist_videos(playlist_url):
    cmd = [YTDLP_BIN, "--flat-playlist", "-J", playlist_url]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(res.stdout)
    return data.get("entries", [])

def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Utilizing Apple Silicon GPU Acceleration (MPS) for WavLM extraction")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Utilizing CUDA GPU Acceleration")
    else:
        device = torch.device("cpu")
        print("Utilizing CPU")
        
    print("Loading Whisper ASR model (small.en)...")
    whisper_model = whisper.load_model("small.en")
    
    print("Loading Text Encoder (all-mpnet-base-v2)...")
    text_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    print("Loading WavLM Feature Extractor (microsoft/wavlm-base-plus)...")
    wavlm_processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()
    
    total_processed = 0
    metadata_records = []
    
    for category_name, info in PLAYLISTS.items():
        playlist_url = info["url"]
        label_idx = info["label"]
        print(f"\n==================================================")
        print(f"📥 Processing Playlist: {category_name.upper()} (Label: {label_idx})")
        print(f"==================================================")
        
        try:
            videos = get_playlist_videos(playlist_url)
        except Exception as e:
            print(f"Error fetching playlist {playlist_url}: {e}")
            continue
            
        print(f"Found {len(videos)} videos. Beginning download & multimodal processing...")
        
        for video in tqdm(videos, desc=f"{category_name}"):
            video_id = video.get("id")
            title = video.get("title", "")
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            if not video_id:
                continue
                
            try:
                audio_path, raw_caption = download_video(video_id, video_url)
                if not os.path.exists(audio_path):
                    continue
                    
                audio_data, sr = sf.read(audio_path)
                if audio_data.ndim > 1:
                    audio_data = audio_data[:, 0]
                audio_fp32 = audio_data.astype(np.float32)
                
                transcription = whisper_model.transcribe(audio_fp32)
                segments = transcription.get("segments", [])
                
                if not segments:
                    continue
                    
                audio_embeds = []
                chunked_text_embeds = []
                cleaned_dialogue_texts = []
                num_segments = len(segments)
                
                for idx, segment in enumerate(segments):
                    seg_text = segment["text"].strip()
                    is_first = (idx < 2)
                    is_last = (idx >= num_segments - 2)
                    
                    if is_broll_or_intro_outro(seg_text, is_first_segment=is_first, is_last_segment=is_last):
                        continue
                        
                    seg_audio = segment_audio(audio_fp32, segment["start"], segment["end"], sr)
                    if seg_audio is None: continue
                        
                    inputs = wavlm_processor(seg_audio, sampling_rate=sr, return_tensors="pt")
                    input_values = inputs.input_values.to(device)
                    with torch.no_grad():
                        outputs = wavlm_model(input_values)
                        a_emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()
                        
                    t_emb = text_model.encode(seg_text, convert_to_tensor=True).cpu()
                    
                    audio_embeds.append(a_emb)
                    chunked_text_embeds.append(t_emb)
                    cleaned_dialogue_texts.append(seg_text)
                    
                if not audio_embeds:
                    for segment in segments:
                        seg_text = segment["text"].strip()
                        if not seg_text: continue
                        seg_audio = segment_audio(audio_fp32, segment["start"], segment["end"], sr)
                        if seg_audio is None: continue
                        inputs = wavlm_processor(seg_audio, sampling_rate=sr, return_tensors="pt")
                        with torch.no_grad():
                            a_emb = wavlm_model(inputs.input_values.to(device)).last_hidden_state.mean(dim=1).squeeze(0).cpu()
                        t_emb = text_model.encode(seg_text, convert_to_tensor=True).cpu()
                        audio_embeds.append(a_emb)
                        chunked_text_embeds.append(t_emb)
                        cleaned_dialogue_texts.append(seg_text)
                        
                audio_tensor = torch.stack(audio_embeds)
                chunked_text_tensor = torch.stack(chunked_text_embeds)
                label_tensor = torch.tensor(label_idx, dtype=torch.long)
                
                output_tensor_path = os.path.join(FEATURES_DIR, f"yt_{video_id}.pt")
                torch.save({
                    "audio_embeds": audio_tensor,
                    "text_embeds": chunked_text_tensor,
                    "label": label_tensor,
                    "label_convention": "project"
                }, output_tensor_path)
                
                metadata_record = {
                    "video_id": video_id,
                    "title": title,
                    "category": category_name,
                    "label": label_idx,
                    "num_dialogue_chunks": len(audio_embeds),
                    "whisper_transcript": " ".join(cleaned_dialogue_texts),
                    "youtube_caption": raw_caption
                }
                metadata_records.append(metadata_record)
                total_processed += 1
                
            except Exception as e:
                print(f"\n⚠️ Error processing video {video_id} ({title}): {e}")
                continue
                
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        for rec in metadata_records:
            f.write(json.dumps(rec) + "\n")
            
    print(f"\n🎉 Successfully converted {total_processed} YouTube videos into multimodal training tensors!")
    print(f"Features saved in: {FEATURES_DIR}/yt_*.pt")
    print(f"Captions & Metadata saved in: {METADATA_FILE}")

if __name__ == "__main__":
    main()
