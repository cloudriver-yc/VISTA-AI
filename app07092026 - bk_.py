import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import librosa
import numpy as np
import whisper
import subprocess
import soundfile as sf
import pandas as pd
import streamlit as st
import transformers
transformers.logging.set_verbosity_error()

from transformers import AutoFeatureExtractor, WavLMModel
from sentence_transformers import SentenceTransformer

# Add src to path so we can import architecture
sys.path.append(os.path.abspath("src"))
from models.architecture import DualTransformerClassifier, EnhancedDualTransformerClassifier


st.set_page_config(
    page_title="VISTA-AI: Dual Transformer CSAT Analyzer",
    page_icon="🎧",
    layout="wide"
)

st.title("🎧 VISTA-AI: Multimodal CSAT & Emotion Analyzer")
st.markdown("Side-by-side evaluation comparing **V1 Baseline (Linear + Mean Pooling)** vs. **V2 Upgraded (MLP ResBlock + [CLS] Token)** on Apple Silicon GPU (`mps`).")

# -----------------
# 1. LOAD MODELS (Cached for UI efficiency)
# -----------------
@st.cache_resource
def load_models():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Whisper ASR
    whisper_model = whisper.load_model("small.en")
    
    # 2. Text Encoder
    text_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    # 3. WavLM Audio Encoder
    wavlm_processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()
    
    # 4. V1 Baseline Model
    model_v1 = DualTransformerClassifier(num_classes=4, audio_dim=768, text_dim=768).to(device)
    v1_path = "models/dual_transformer_v1_weights.pt" if os.path.exists("models/dual_transformer_v1_weights.pt") else "models/dual_transformer_weights.pt"
    if os.path.exists(v1_path):
        model_v1.load_state_dict(torch.load(v1_path, map_location=device))
        model_v1.eval()
        
    # 5. V2 Upgraded Model
    model_v2 = EnhancedDualTransformerClassifier(num_classes=4, audio_dim=768, text_dim=768).to(device)
    v2_path = "models/dual_transformer_v2_weights.pt" if os.path.exists("models/dual_transformer_v2_weights.pt") else "models/dual_transformer_weights.pt"
    if os.path.exists(v2_path):
        model_v2.load_state_dict(torch.load(v2_path, map_location=device))
        model_v2.eval()
        
    return device, whisper_model, text_model, wavlm_processor, wavlm_model, model_v1, model_v2

device, whisper_model, text_model, wavlm_processor, wavlm_model, model_v1, model_v2 = load_models()

# -----------------
# 2. UI INPUT
# -----------------
with st.container(border=True):
    youtube_url = st.text_input("Enter YouTube Video URL:", value="https://youtu.be/gD7xQGXpSBg")
    run_button = st.button("Analyze Audio with Both Models", type="primary")

def get_video_id(url):
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")
    elif "youtube.com" in parsed.netloc:
        return urllib.parse.parse_qs(parsed.query).get("v", [None])[0]
    return "unknown_video"

if run_button and youtube_url:
    video_id = get_video_id(youtube_url)
    if not video_id:
        st.error("Invalid YouTube URL.")
        st.stop()
        
    tmp_audio = f"data/audio/tmp/{video_id}.wav"
    os.makedirs("data/audio/tmp", exist_ok=True)
    
    # -----------------
    # 3. AUDIO INGESTION & ASR
    # -----------------
    with st.spinner("Downloading YouTube Audio..."):
        if not os.path.exists(tmp_audio):
            cmd = [
                "yt-dlp",
                "-x", "--audio-format", "wav",
                "--postprocessor-args", "-ar 16000 -ac 1",
                "-o", tmp_audio.replace(".wav", ""),
                youtube_url
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError:
                st.error("YouTube blocked automated download (HTTP 429). Please run manual download with browser cookies.")
                st.stop()
        else:
            st.info("Audio cached locally.")
            
    with st.spinner("Transcribing with Whisper & extracting WavLM prosody + MPNet text embeddings..."):
        audio_data, sr = sf.read(tmp_audio)
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]
            
        audio_duration = len(audio_data) / sr
        customer_audio_fp32 = audio_data.astype(np.float32)
        transcription = whisper_model.transcribe(customer_audio_fp32)
        
        audio_embeds = []
        chunked_text_embeds = []
        full_transcript = ""
        segment_details = []
        
        for idx, segment in enumerate(transcription["segments"]):
            seg_text = segment["text"].strip()
            if not seg_text: continue
            
            full_transcript += f"{seg_text} "
            start_sample = int(segment["start"] * sr)
            end_sample = int(segment["end"] * sr)
            seg_audio = customer_audio_fp32[start_sample:end_sample]
            
            if len(seg_audio) < 160: continue
            
            # WavLM Acoustic Prosody
            inputs = wavlm_processor(seg_audio, sampling_rate=sr, return_tensors="pt")
            input_values = inputs.input_values.to(device)
            with torch.no_grad():
                outputs = wavlm_model(input_values)
                a_emb = outputs.last_hidden_state.mean(dim=1).squeeze(0) # (768,)
                
            # Text Semantics
            t_emb = text_model.encode(seg_text, convert_to_tensor=True).to(device)
            
            audio_embeds.append(a_emb)
            chunked_text_embeds.append(t_emb)
            segment_details.append({
                "#": idx + 1,
                "Start": f"{segment['start']:.1f}s",
                "End": f"{segment['end']:.1f}s",
                "Duration": f"{segment['end'] - segment['start']:.1f}s",
                "Text": seg_text,
                "Audio Norm (||a||)": f"{a_emb.norm().item():.2f}",
                "Text Norm (||t||)": f"{t_emb.norm().item():.2f}"
            })
            
        st.success(f"Extracted {len(audio_embeds)} dialogue segments (Total duration: {audio_duration:.1f}s).")
        
    with st.expander("📝 View Full Dialogue Transcript", expanded=False):
        st.write(full_transcript)
        if segment_details:
            st.dataframe(pd.DataFrame(segment_details), use_container_width=True)
        
    # -----------------
    # 4. SIDE-BY-SIDE INFERENCE
    # -----------------
    with st.spinner("Executing Dual Model Inference (V1 Baseline & V2 Upgraded)..."):
        if not audio_embeds:
            st.error("No valid audio segments detected in the recording.")
            st.stop()
            
        audio_tensor = torch.stack(audio_embeds).unsqueeze(0).to(device)       # (1, S, 768)
        chunked_text_tensor = torch.stack(chunked_text_embeds).unsqueeze(0).to(device) # (1, S, 768)
        
        # V1 Inference
        t0 = time.perf_counter()
        with torch.no_grad():
            logits_v1, _, _ = model_v1(chunked_text_tensor, audio_tensor)
            probs_v1 = torch.softmax(logits_v1, dim=1).squeeze(0)
        t_v1 = (time.perf_counter() - t0) * 1000
        
        # V2 Inference
        t0 = time.perf_counter()
        with torch.no_grad():
            logits_v2, _, _ = model_v2(chunked_text_tensor, audio_tensor)
            probs_v2 = torch.softmax(logits_v2, dim=1).squeeze(0)
        t_v2 = (time.perf_counter() - t0) * 1000
        
        classes = ["Very Unsatisfied", "Unsatisfied", "Satisfied", "Very Satisfied"]
        
        pred_v1_idx = torch.argmax(probs_v1).item()
        pred_v2_idx = torch.argmax(probs_v2).item()
        
        pred_v1 = classes[pred_v1_idx]
        pred_v2 = classes[pred_v2_idx]
        
        conf_v1 = probs_v1[pred_v1_idx].item() * 100
        conf_v2 = probs_v2[pred_v2_idx].item() * 100
        
    # -----------------
    # 5. COMPARATIVE RESULTS DISPLAY
    # -----------------
    st.markdown("### ⚖️ Side-by-Side Model Prediction Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("#### 🔹 V1 Baseline (Linear + Mean Pooling)")
            st.metric(label="Predicted CSAT", value=pred_v1)
            st.metric(label="Confidence", value=f"{conf_v1:.2f}%")
            st.caption(f"⚡ Latency: {t_v1:.2f}ms on Apple Silicon MPS")
            
    with col2:
        with st.container(border=True):
            st.markdown("#### 🚀 V2 Upgraded (MLP ResBlock + [CLS] Token)")
            st.metric(label="Predicted CSAT", value=pred_v2)
            st.metric(label="Confidence", value=f"{conf_v2:.2f}%", delta=f"{conf_v2 - conf_v1:+.2f}% vs V1")
            st.caption(f"⚡ Latency: {t_v2:.2f}ms on Apple Silicon MPS")
            
    st.markdown("### 📊 Probability Distribution Comparison")
    df_compare = pd.DataFrame({
        "V1 Baseline (%)": [p.item() * 100 for p in probs_v1],
        "V2 Upgraded (%)": [p.item() * 100 for p in probs_v2],
        "Category": classes
    }).set_index("Category")
    
    st.bar_chart(df_compare)
    
    # -----------------
    # 6. DEEP DEBUG DIAGNOSTICS & TELEMETRY
    # -----------------
    with st.expander("🔬 Deep Multimodal Debug & Diagnostics", expanded=True):
        st.markdown("#### 1. Prediction Delta Table")
        df_deltas = pd.DataFrame({
            "CSAT Category": classes,
            "V1 Baseline (%)": [f"{p.item() * 100:.2f}%" for p in probs_v1],
            "V2 Upgraded (%)": [f"{p.item() * 100:.2f}%" for p in probs_v2],
            "Delta (V2 - V1)": [f"{(p2.item() - p1.item()) * 100:+.2f}%" for p1, p2 in zip(probs_v1, probs_v2)]
        })
        st.table(df_deltas)
        
        st.markdown("#### 2. Raw Tensor Logits & Geometry")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.write("**V1 Raw Logits:**", [round(x, 4) for x in logits_v1[0].tolist()])
            st.write("**Audio Tensor Shape:**", tuple(audio_tensor.shape))
        with m_col2:
            st.write("**V2 Raw Logits:**", [round(x, 4) for x in logits_v2[0].tolist()])
            st.write("**Text Tensor Shape:**", tuple(chunked_text_tensor.shape))
        with m_col3:
            st.write("**Hardware Backend:**", str(device))
            st.write("**Model Consensus:**", "✅ Agreement" if pred_v1 == pred_v2 else "⚠️ Divergence")
            
        st.markdown("#### 3. Segment Embedding Norm Breakdown")
        if segment_details:
            st.dataframe(pd.DataFrame(segment_details), use_container_width=True)


