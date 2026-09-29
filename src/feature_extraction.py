"""
Shared feature extraction: Whisper ASR -> per-segment WavLM (acoustic) + MPNet (text) embeddings.
Used by scripts/04_extract_features.py (bulk) and scripts/ingest_upload.py (single uploaded calls).
"""
import torch
import soundfile as sf
import numpy as np
from audio_utils import segment_audio

LABEL_MAP = {
    # Canonical labels (0 to 3)
    "very_unsatisfied": 0,
    "unsatisfied": 1,
    "satisfied": 2,
    "very_satisfied": 3,
    # Legacy aliases
    "urgent_follow_up": 3,
    "at_risk_dissatisfied": 1,
    "standard_resolved": 2,
    "promoter_delighted": 0
}

def extract_features_from_audio(audio_path, sr_target=16000, whisper_model=None, wavlm_processor=None, wavlm_model=None, text_model=None, device="cpu", channel="mix"):
    """
    Transcribes audio with Whisper, segments dialogue turns, and extracts 768-d WavLM + MPNet features.
    channel="mix" sums stereo to mono (both speakers, like real calls); channel="customer" keeps ch0 only.
    """
    audio_data, sr = sf.read(audio_path)
    if audio_data.ndim > 1:
        if channel == "mix":
            # Speakers rarely overlap, so summing keeps each speaker at its natural level
            customer_audio = np.clip(audio_data.sum(axis=1), -1.0, 1.0)
        else:
            customer_audio = audio_data[:, 0]
    else:
        customer_audio = audio_data
        
    customer_audio_fp32 = customer_audio.astype(np.float32)
    transcription = whisper_model.transcribe(customer_audio_fp32)
    
    audio_embeds = []
    chunked_text_embeds = []
    
    for segment in transcription["segments"]:
        seg_text = segment["text"].strip()
        if not seg_text:
            continue
            
        seg_audio = segment_audio(customer_audio_fp32, segment["start"], segment["end"], sr)
        if seg_audio is None:
            continue
            
        # WavLM Acoustic Prosody
        inputs = wavlm_processor(seg_audio, sampling_rate=sr, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            outputs = wavlm_model(input_values)
            a_emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()
            
        # Text Semantics
        t_emb = text_model.encode(seg_text, convert_to_tensor=True).cpu()
        
        audio_embeds.append(a_emb)
        chunked_text_embeds.append(t_emb)
        
    if not audio_embeds:
        audio_embeds.append(torch.zeros(768))
        chunked_text_embeds.append(torch.zeros(768))
        
    return torch.stack(audio_embeds), torch.stack(chunked_text_embeds)


def get_device():
    """Apple Silicon GPU (MPS) first, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_extractors(device):
    """Loads Whisper small.en, WavLM-base-plus and all-mpnet-base-v2 (keyword args for extract_features_from_audio)."""
    import whisper
    from transformers import AutoFeatureExtractor, WavLMModel
    from sentence_transformers import SentenceTransformer

    print("Loading Text Encoder (all-mpnet-base-v2 for 768-d embeddings)...")
    text_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    print("Loading WavLM Model (microsoft/wavlm-base-plus for 768-d acoustic prosody)...")
    wavlm_processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()
    print("Loading Whisper ASR model (small.en)...")
    whisper_model = whisper.load_model("small.en")
    return dict(whisper_model=whisper_model, wavlm_processor=wavlm_processor,
                wavlm_model=wavlm_model, text_model=text_model, device=device)
