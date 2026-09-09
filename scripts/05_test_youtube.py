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
import transformers
transformers.logging.set_verbosity_error()

from transformers import AutoFeatureExtractor, WavLMModel
from sentence_transformers import SentenceTransformer

# Add src to path so we can import architecture
sys.path.append(os.path.abspath("src"))
from models.architecture import DualTransformerClassifier, EnhancedDualTransformerClassifier
from models.explainer import MultimodalExplainer



def main():
    youtube_url = "https://youtu.be/gD7xQGXpSBg?si=Ok1BDZTkzom_K__6"
    tmp_audio = "data/audio/tmp/youtube.wav"
    
    os.makedirs("data/audio/tmp", exist_ok=True)
    
    if os.path.exists(tmp_audio):
        print(f"1. Audio already exists at {tmp_audio}, skipping download.")
    else:
        print("1. Downloading YouTube Audio...")
        cmd = [
            "yt-dlp",
            "-x", "--audio-format", "wav",
            "--postprocessor-args", "-ar 16000 -ac 1",
            "-o", tmp_audio.replace(".wav", ""),
            youtube_url
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print("\n[ERROR] YouTube blocked the download (HTTP 429 / Bot Protection).")
            print("Try running this manual command in your terminal first:")
            print(f"yt-dlp --cookies-from-browser chrome -x --audio-format wav -o data/audio/tmp/youtube {youtube_url}\n")
            sys.exit(1)
    
    # Device Selection (Apple Silicon GPU MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Utilizing Apple Silicon GPU Acceleration (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Utilizing CUDA GPU Acceleration")
    else:
        device = torch.device("cpu")
        print("Utilizing CPU")

    print("2. Loading Feature Extractors (Whisper + WavLM + MPNet)...")
    whisper_model = whisper.load_model("small.en")
    text_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    wavlm_processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()
    
    # Load Both Models
    print("   • Loading V1 Baseline (Linear Projections + Mean Pooling)...")
    model_v1 = DualTransformerClassifier(num_classes=4, audio_dim=768, text_dim=768).to(device)
    v1_path = "models/dual_transformer_v1_weights.pt" if os.path.exists("models/dual_transformer_v1_weights.pt") else "models/dual_transformer_weights.pt"
    model_v1.load_state_dict(torch.load(v1_path, map_location=device))
    model_v1.eval()
    
    print("   • Loading V2 Upgraded (MLP ResBlock + [CLS] Token)...")
    model_v2 = EnhancedDualTransformerClassifier(num_classes=4, audio_dim=768, text_dim=768).to(device)
    v2_path = "models/dual_transformer_v2_weights.pt" if os.path.exists("models/dual_transformer_v2_weights.pt") else "models/dual_transformer_weights.pt"
    model_v2.load_state_dict(torch.load(v2_path, map_location=device))
    model_v2.eval()
    
    print("3. Transcribing with Whisper & Extracting Dynamic Chunks...")
    audio_data, sr = librosa.load(tmp_audio, sr=16000)
    audio_duration = len(audio_data) / sr
    audio_fp32 = audio_data.astype(np.float32)
    transcription = whisper_model.transcribe(audio_fp32)
    
    print("\n" + "=" * 80)
    print("📝 WHISPER TRANSCRIPT")
    print("=" * 80)
    print(transcription["text"].strip())
    print("=" * 80 + "\n")
    
    audio_embeds = []
    chunked_text_embeds = []
    chunk_debug_stats = []
    
    for idx, segment in enumerate(transcription["segments"]):
        seg_text = segment["text"].strip()
        if not seg_text: continue
            
        start_sample = int(segment["start"] * sr)
        end_sample = int(segment["end"] * sr)
        seg_audio = audio_fp32[start_sample:end_sample]
        
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
        
        chunk_debug_stats.append({
            "chunk_idx": idx + 1,
            "start": segment["start"],
            "end": segment["end"],
            "duration": segment["end"] - segment["start"],
            "text": seg_text,
            "audio_norm": a_emb.norm().item(),
            "text_norm": t_emb.norm().item()
        })
        
    if not audio_embeds:
        print("No valid segments found!")
        return
        
    audio_tensor = torch.stack(audio_embeds).unsqueeze(0)       # (1, S, 768)
    chunked_text_tensor = torch.stack(chunked_text_embeds).unsqueeze(0) # (1, S, 768)
    num_chunks = audio_tensor.size(1)
    
    print(f"📊 Extracted {num_chunks} dynamic dialogue chunks (Total Audio Duration: {audio_duration:.2f}s).")
    
    # 4. Run Both Models Concurrently with Benchmarking
    print("\n4. Running Side-by-Side Inference (V1 Baseline vs. V2 Upgraded)...")
    explainer = MultimodalExplainer()
    
    # V1 Inference with XAI
    t0 = time.perf_counter()
    with torch.no_grad():
        xai_v1 = model_v1(chunked_text_tensor, audio_tensor, return_xai=True)
        logits_v1 = xai_v1["logits"]
        probs_v1 = xai_v1["probabilities"][0]
    t_v1 = (time.perf_counter() - t0) * 1000
    
    # V2 Inference with XAI
    t0 = time.perf_counter()
    with torch.no_grad():
        xai_v2 = model_v2(chunked_text_tensor, audio_tensor, return_xai=True)
        logits_v2 = xai_v2["logits"]
        probs_v2 = xai_v2["probabilities"][0]
    t_v2 = (time.perf_counter() - t0) * 1000
    
    classes = ["Very Unsatisfied", "Unsatisfied", "Satisfied", "Very Satisfied"]
    pred_v1_idx = xai_v1["predicted_class"]
    pred_v2_idx = xai_v2["predicted_class"]
    
    pred_v1 = classes[pred_v1_idx]
    pred_v2 = classes[pred_v2_idx]
    
    # Generate XAI explanations
    expl_v1 = explainer.explain(xai_v1, chunk_debug_stats, top_k=3)
    expl_v2 = explainer.explain(xai_v2, chunk_debug_stats, top_k=3)
    
    # --- PRINT COMPARATIVE RESULTS TABLE ---
    print("\n" + "=" * 80)
    print("⚖️  SIDE-BY-SIDE MODEL PREDICTION COMPARISON")
    print("=" * 80)
    print(f"{'CSAT Category':<22} | {'V1 Baseline Prob':<18} | {'V2 Upgraded Prob':<18} | {'Delta (V2 - V1)'}")
    print("-" * 80)
    for c, p1, p2 in zip(classes, probs_v1, probs_v2):
        val1 = p1.item() * 100
        val2 = p2.item() * 100
        delta = val2 - val1
        sign = "+" if delta >= 0 else ""
        print(f"{c:<22} | {val1:15.2f}% | {val2:15.2f}% | {sign}{delta:6.2f}%")
        
    print("-" * 80)
    print(f"{'Predicted Category:':<22} | {pred_v1:<18} | {pred_v2:<18} | {'MATCH ✅' if pred_v1 == pred_v2 else 'DIFFERENT ⚠️'}")
    print(f"{'Inference Latency:':<22} | {t_v1:15.2f}ms | {t_v2:15.2f}ms | {t_v2 - t_v1:+6.2f}ms")
    print("=" * 80)
    
    # --- EXPLAINABLE AI (XAI) DECISION ATTRIBUTION ---
    print("\n" + "=" * 80)
    print("🔍 EXPLAINABLE AI (XAI): WHY DID THE MODEL MAKE THIS DECISION?")
    print("=" * 80)
    print(f"• Modality Attribution (V2):    Text Semantics: {expl_v2['modality_attribution']['text_influence_pct']}% | Acoustic Prosody: {expl_v2['modality_attribution']['audio_influence_pct']}%")
    print(f"• Primary Decision Driver:      {expl_v2['modality_attribution']['dominant_modality']}")
    print("\nTop Pivotal Dialogue Turns (Highest Decision Influence):")
    print(f"{'Turn':<5} | {'Timestamp':<15} | {'Saliency':<10} | {'Tone Diagnosis':<30} | {'Text Content'}")
    print("-" * 80)
    for pt in expl_v2["top_pivotal_turns"]:
        print(f"{pt['turn_index']:<5} | {pt['timestamp']:<15} | {pt['saliency_pct']:6.2f}%    | {pt['tone_diagnosis']:<30} | {pt['text'][:45]}")
        
    print("\n📖 Executive Decision Audit:")
    print(f"  {expl_v2['narrative_rationale']}")
    print("=" * 80)
    
    # --- DEEP MULTIMODAL DEBUG INFO ---
    print("\n" + "=" * 80)
    print("🔬 DEEP MULTIMODAL DIAGNOSTICS & DEBUG TELEMETRY")
    print("=" * 80)
    print(f"• Hardware Device:               {device}")
    print(f"• Total Conversation Chunks:     {num_chunks}")
    print(f"• Audio Tensor Dimensions:       {tuple(audio_tensor.shape)}")
    print(f"• Text Tensor Dimensions:        {tuple(chunked_text_tensor.shape)}")
    print(f"• Mean Audio Embedding Norm:     {np.mean([s['audio_norm'] for s in chunk_debug_stats]):.4f}")
    print(f"• Mean Text Embedding Norm:      {np.mean([s['text_norm'] for s in chunk_debug_stats]):.4f}")
    print(f"• V1 Raw Output Logits:          {[round(x, 4) for x in logits_v1[0].tolist()]}")
    print(f"• V2 Raw Output Logits:          {[round(x, 4) for x in logits_v2[0].tolist()]}")
    print(f"• V1 Prediction Confidence:      {probs_v1[pred_v1_idx].item()*100:.2f}% ({pred_v1})")
    print(f"• V2 Prediction Confidence:      {probs_v2[pred_v2_idx].item()*100:.2f}% ({pred_v2})")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

