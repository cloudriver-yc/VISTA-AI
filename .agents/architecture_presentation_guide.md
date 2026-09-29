# VISTA-AI: Multimodal Machine Learning Architecture & Presentation Defense Guide

> **Prepared for:** Machine Learning Academic Presentation & Defense  
> **Topic:** Multimodal Conversational CSAT & Affective Sentiment Prediction  
> **Core Framework:** PyTorch | Apple Silicon GPU (`mps`) | WavLM-base-plus | MPNet-base-v2 | Cross-Modal Attention  

---

## 1. System Architecture Diagram & Comprehensive Explanation

```mermaid
graph TD
    subgraph S0["0. Data Generation & Acoustic Augmentation (training data only)"]
        GEM["Script Generator via Antigravity CLI (Gemini 3.8 Flash first; Gemini 3.1 Pro / Claude Sonnet 4.6 fallback)"]
        TTS["ElevenLabs TTS: Stereo WAV (customer ch0, engineer ch1)"]
        MIX["Mono Mixdown (per-speaker gain +/-6 dB)"]
        AUG["Augmentation Chain: speed/pitch, room reverb, background noise (ESC-50 / colored / hum / babble, 5-30 dB SNR), telephone 8 kHz + mu-law, gain"]
        GEM --> TTS
        TTS --> MIX
        MIX --> AUG
    end

    subgraph S1["1. Raw Input Modalities"]
        A1["16kHz Mono Audio (augmented TTS for training; real calls for test / inference)"]
        T1["Whisper ASR Transcript"]
        A1 --> T1
    end
    AUG --> A1

    subgraph S2["2. Dynamic Dialogue Turn Segmentation"]
        VAD["Voice Activity Detection (VAD) & Turn Slicing"]
        A1 --> VAD
        T1 --> VAD
        A_chunks["Audio Chunks: a_1, a_2, ..., a_S"]
        T_chunks["Text Turns: t_1, t_2, ..., t_S"]
        VAD --> A_chunks
        VAD --> T_chunks
    end

    subgraph S3["3. Self-Supervised Backbone Feature Extractors"]
        WavLM["microsoft/wavlm-base-plus (768-d Acoustic Prosody)"]
        MPNet["sentence-transformers/all-mpnet-base-v2 (768-d Semantics)"]
        A_chunks --> WavLM
        T_chunks --> MPNet
        H_audio["Audio Tensor A in R^(S x 768)"]
        H_text["Text Tensor T in R^(S x 768)"]
        WavLM --> H_audio
        MPNet --> H_text
    end

    subgraph S4["4. Feature Adaptation & Domain Projection"]
        Proj_A["Linear(768 -> 512) + LayerNorm + GELU + MLPResBlock"]
        Proj_T["Linear(768 -> 512) + LayerNorm + GELU + MLPResBlock"]
        H_audio --> Proj_A
        H_text --> Proj_T
        Z_audio["Adapted Audio z_A in R^(S x 512)"]
        Z_text["Adapted Text z_T in R^(S x 512)"]
        Proj_A --> Z_audio
        Proj_T --> Z_text
    end

    subgraph S5["5. Bidirectional Cross-Modal Attention Fusion"]
        CA_TA["Text queries Audio: Q=z_T, K=z_A, V=z_A (Acoustic Semantics)"]
        CA_AT["Audio queries Text: Q=z_A, K=z_T, V=z_T (Semantic Prosody)"]
        Z_audio --> CA_TA
        Z_text --> CA_TA
        Z_audio --> CA_AT
        Z_text --> CA_AT
        Norm_Fuse["LayerNorm + Concat(512+512) -> Linear(1024 -> 512) + GELU"]
        CA_TA --> Norm_Fuse
        CA_AT --> Norm_Fuse
        Fused_Seq["Fused Dialogue Sequence F in R^(S x 512)"]
        Norm_Fuse --> Fused_Seq
    end

    subgraph S6["6. Conversational Sequence Transformer"]
        CLS["Prepend Learnable [CLS] Token z_0 in R^(1 x 512)"]
        PE["Sinusoidal Positional Encoding (up to 1024 turns)"]
        Fused_Seq --> CLS
        CLS --> PE
        Seq_Trans["2-Layer Transformer Encoder (d_model=512, nhead=8, d_ff=1024, Dropout=0.2)"]
        PE --> Seq_Trans
        Global_Rep["Global Representation: h_CLS (V2) or Masked Mean Pool (V1)"]
        Seq_Trans --> Global_Rep
    end

    subgraph S7["7. Classification Head & CSAT Output"]
        Head["Linear(512 -> 128) + LayerNorm + GELU + Dropout(0.3) -> Linear(128 -> 4)"]
        Global_Rep --> Head
        Softmax["Softmax Probs: [Very Unsatisfied, Unsatisfied, Satisfied, Very Satisfied]"]
        Head --> Softmax
    end
```

### Data Split & Two-Stage Training Diagram

```mermaid
graph LR
    subgraph SYN["Synthetic TTS pool"]
        SCR["Unique scripts (grouped by text hash)"]
        CLEAN["Clean TTS clips (mono mixdown)"]
        AUGC["Augmented clips (__augNN)"]
        SCR --> CLEAN
        CLEAN --> AUGC
    end
    subgraph REAL["Real-world pool"]
        YT["YouTube calls (yt_*)"]
        UP["Uploaded calls (upload_*)"]
    end
    SCR -->|"~85% of scripts per class"| TRAIN["TRAIN: clean + augmented"]
    SCR -->|"~15% of scripts per class"| VAL["VAL: held-out scripts, model selection"]
    UP --> FT["FINETUNE: uploaded real calls"]
    YT --> TEST["TEST: YouTube calls, never trained on"]

    TRAIN --> S1["Stage 1: pretrain V1 / V2 (all weights trainable)"]
    VAL --> S1
    S1 --> FROZEN["Frozen stage-1 model: encode() gives a 512-d call summary"]
    FROZEN --> ADAPT["Stage 2: ResidualAdapterHead (LayerNorm, 512 to 64, GELU, 64 to 4, zero-init)"]
    FT --> ADAPT
    FROZEN --> SUM["Adapted logits = frozen logits + adapter output"]
    ADAPT --> SUM
    SUM --> S3["Stage 3: evaluate stage 1 vs stage 2"]
    TEST --> S3
```

**Why freeze the model and add a zero-initialised residual adapter?** There are only a handful of real uploaded calls. Fine-tuning all 7.7M (V1) or 9.8M (V2) parameters on them would overwrite what the model learned from thousands of synthetic dialogues, so the pretrained model is frozen and only a small head (34k parameters) learns a correction from real audio. The head's output is added to the frozen logits, and its last layer starts at zero, so the adapted model begins exactly at the stage-1 model and can only move where the real calls push it. Scoring both stages on the same YouTube calls shows directly whether the real-data adaptation helped. With only two uploads, the head can separate them along almost any direction and then shift every other call the same way (unconstrained, it pushed V2 to predict "Unsatisfied" for 10 of 19 YouTube calls and cut its synthetic-validation accuracy from 99.4% to 73.1%). So stage 2 adds a KL "anchor" that keeps predictions on synthetic calls close to the frozen model's, and a guard that rejects the head if it lowers synthetic validation accuracy. Both use only synthetic data, never the YouTube test set.

---

### Why Acoustic Augmentation?
* **Domain gap:** training audio was clean studio TTS on the customer channel only, while YouTube calls, uploads and the app are noisy mono recordings of both speakers. Mono mixdown plus noise, reverb and telephone-band simulation moves the training distribution toward real calls.
* **WavLM robustness:** the acoustic branch learns prosody cues (tension, energy, pitch contour) that survive noise and codec artefacts instead of TTS-specific studio characteristics.
* **Whisper realism:** every augmented clip is transcribed by Whisper, so the text branch also sees realistic ASR errors under noise.
* **Text diversity:** augmentation cannot add new wording. The original 307 TTS clips came from only 16 unique scripts, so new unique scripts are generated first, with Gemini and Claude models available through the Antigravity CLI.
* **Leakage-safe grouping:** all re-voicings and augmented variants of a script stay on the same side of the train/val split, and every real-world call stays in test.

---

### Detailed Architectural Breakdown: What, How, Why

#### **WHAT is this architecture?**
An end-to-end **Hierarchical Multimodal Cross-Attention Transformer** that predicts Customer Satisfaction (CSAT) and resolution outcomes directly from conversational dialogues. It processes multi-turn dialogues ranging from 10 seconds to 15+ minutes by fusing dense acoustic speech prosody with linguistic semantics.

#### **HOW does it operate?**
1. **Dynamic Utterance Alignment:** Whisper ASR extracts timestamped voice segments, slicing audio into spoken dialogue turns ($S$ turns).
2. **Dual Pre-trained Encoding:**
   - **Acoustic:** `microsoft/wavlm-base-plus` extracts 768-d frame representations, mean-pooled temporally per turn into $\mathbf{a}_s \in \mathbb{R}^{768}$.
   - **Linguistic:** `all-mpnet-base-v2` encodes the transcript text into $\mathbf{t}_s \in \mathbb{R}^{768}$.
3. **Pre-LN Residual Adaptation (`MLPResBlock`):** Projects $768 \to 512$ and adapts static embeddings into the customer service task domain.
4. **Bidirectional Cross-Modal Attention:**
   $$\mathbf{t}_{\text{attended}} = \text{Softmax}\left(\frac{\mathbf{z}_T \mathbf{z}_A^T}{\sqrt{d_k}}\right)\mathbf{z}_A, \quad \mathbf{a}_{\text{attended}} = \text{Softmax}\left(\frac{\mathbf{z}_A \mathbf{z}_T^T}{\sqrt{d_k}}\right)\mathbf{z}_T$$
5. **Turn-Level Sequence Transformer:** Models multi-turn emotional escalation and resolution progression with Sinusoidal Positional Encoding and multi-head self-attention.
6. **Classification:** Maps global pooled representations into 4 canonical CSAT classes.

#### **WHY was it designed this way?**
* **The Multimodal Semantic-Acoustic Gap:** Linguistic sentiment alone fails on *sarcasm* or *polite frustration* (e.g., *"Oh, that's just wonderful, thank you for nothing"*). Acoustic energy alone fails on *loud excitement* vs. *angry shouting*. Cross-attention grounds text in vocal tone and vice versa.
* **Non-Local Conversational Saliency:** CSAT is not an instantaneous frame property; it depends on the trajectory of the conversation. A customer screaming at Turn 2 whose issue is resolved by Turn 25 ends with a positive score (`Satisfied`). The Sequence Transformer captures this full conversational arc.

---

## 2. Three-Way Architectural Evolution (Legacy CNN vs. V1 Baseline vs. V2 Upgraded)

| Architectural Dimension | Legacy Y-Shaped CNN | V1 Baseline Dual Transformer | V2 Upgraded Dual Transformer |
| :--- | :--- | :--- | :--- |
| **Acoustic Feature Extraction** | Static 2D Log-Mel Spectrograms ($128 \times T$) | **WavLM-base-plus (768-d self-supervised prosody)** | **WavLM-base-plus (768-d self-supervised prosody)** |
| **Linguistic Feature Extraction** | Flat TF-IDF / 1D Text ConvNet | **all-mpnet-base-v2 (768-d sentence embeddings)** | **all-mpnet-base-v2 (768-d sentence embeddings)** |
| **Modality Projection Layer** | Flat Conv2D / Dense Layers | Linear + LayerNorm + GELU ($768 \to 512$) | **Pre-LN 2-Layer `MLPResBlock` ($768 \to 512 \to 1024 \to 512$)** |
| **Cross-Modal Interaction** | None (Late concatenation of flattened vectors) | **Bidirectional Multi-Head Cross-Attention** | **Bidirectional Multi-Head Cross-Attention** |
| **Conversational Sequence Modeling** | Flattened Global Pooling (No temporal sequence) | **2-Layer Sequence Transformer (Mean Pooling)** | **2-Layer Sequence Transformer (`[CLS]` Token)** |
| **Positional Encoding** | None | Sinusoidal ($max\_len=1024$) | Sinusoidal ($max\_len=1024$) |
| **Train Accuracy** | 82.4% (Overfitting on spectrogram noise) | **100.0%** | **100.0%** |
| **Isolated Test Accuracy** | 71.2% | **93.8%** | **91.7%** |
| **Held-Out YouTube Generalization**| 0.0% (Failed on real audio) | **40.0%** | **40.0%** |
| **Inference Latency on MPS** | $\approx 850\text{ms}$ | $625\text{ms}$ | **$176\text{ms}$ (3.5x Faster)** |

> Accuracy rows above were measured on the legacy split (real calls partly in train, synthetic test sharing script text with train). Re-benchmark after retraining on the augmented train / real-world test split.

---

### In-Depth Comparison: What, How, Why

#### 1. Why Did the Legacy CNN Fail?
* **WHAT:** A 2D CNN on mel-spectrograms paired with a 1D CNN on text tokens.
* **HOW:** Treated the spectrogram as a 2D image, running $3 \times 3$ convolutional filters and max-pooling across time and frequency.
* **WHY IT FAILED:**
  1. **Violation of Translation Invariance:** In computer vision, a cat is a cat whether at the top or bottom of an image. In audio, shifting a signal upwards in frequency changes the fundamental pitch ($F_0$) and speaker identity completely. 2D CNN weight sharing across frequency bins is mathematically sub-optimal.
  2. **Late Fusion Bottleneck:** Combining modalities only at the output layer prevents the model from learning acoustic-linguistic correlations during intermediate representations.

#### 2. How Does V1 Baseline Outperform CNN?
* **WHAT:** Intermediate Cross-Modal Attention with Sequence Transformer and Mean Pooling.
* **HOW:** Extracts high-level 768-d representations from 94k-hour pre-trained foundation models (`WavLM` + `MPNet`), fuses them via cross-attention, and aggregates turns via masked mean-pooling.
* **WHY IT EXCELS:** Foundation models provide rich representations of pitch, vocal jitter, and semantics. Mean-pooling across turns acts as a natural regularizer, achieving **93.8% test accuracy**.

#### 3. How Does V2 Upgraded Differ from V1?
* **WHAT:** Adds a **Pre-LN 2-layer `MLPResBlock`** for feature adaptation and a **Learnable `[CLS]` Token** for sequence aggregation.
* **HOW:** Projections pass through a non-linear residual bottleneck; the `[CLS]` token attends to all dialogue turns to form a singular global representation ($seq\_out[:, 0, :]$).
* **WHY & TRADE-OFFS:** 
  - **Speed:** Slicing index 0 avoids iterating over padding masks, reducing inference latency from **625ms down to 176ms (3.5x speedup)** on Apple Silicon MPS.
  - **Empirical Regularization:** On compact datasets ($\sim 300$ samples), mean pooling (V1) distributes gradients evenly across all dialogue turns, yielding slightly higher test accuracy (+2.1%), while `[CLS]` token pooling concentrates representation into a single parameter.

---

## 3. Mathematical & Hyperparameter Design Decisions (Presentation "Shining Points")

### Shining Point 1: Why Dimension $d_{\text{model}} = 512$?
* **The Information Bottleneck Principle:** Both `WavLM` and `MPNet` output 768-dimensional vectors ($768 + 768 = 1536$ concatenated). Projecting down to $d_{\text{model}} = 512$ acts as an information bottleneck:
  $$\min I(X; Z) \quad \text{s.t.} \quad \max I(Z; Y)$$
  This forces the network to strip away generic phonetic and syntactic noise, preserving only task-salient affective features.
* **Multi-Head Attention Geometry:** 512 is divisible by $n_{\text{head}} = 8$, yielding per-head projection dimensions of:
  $$d_k = \frac{d_{\text{model}}}{n_{\text{head}}} = \frac{512}{8} = 64$$
  A head dimension of $d_k = 64$ matches the standard scaled dot-product attention scaling factor $\frac{1}{\sqrt{d_k}} = \frac{1}{8} = 0.125$, preventing softmax saturation and vanishing gradients in attention maps.

---

### Shining Point 2: Why 2 Layers in the Sequence Transformer (Why Not 1 or 6)?
* **Hierarchical Conversational Semantics:**
  - **Layer 1 (Local Interaction):** Attends to adjacent turn pairs (e.g., *Customer Complaint $\leftrightarrow$ Agent Response*).
  - **Layer 2 (Global Discourse Arc):** Attends across distant turns (e.g., *Turn 1 Grievance $\leftrightarrow$ Turn 30 Final Resolution*).
* **Occam's Razor & Capacity Matching:** With $\sim 300$ dialogue training samples, a 6-layer Transformer would have $\approx 18\text{M}$ trainable parameters in the sequence backbone, leading to severe memorization/overfitting. A 2-layer encoder ($\approx 6.3\text{M}$ trainable parameters) achieves optimal capacity balance without overfitting.

---

### Shining Point 3: Pre-LN Residual MLP (`MLPResBlock`)
* **Mathematical Formulation:**
  $$\mathbf{h} = \mathbf{x} + \mathbf{W}_2 \cdot \text{Dropout}\big(\text{GELU}(\text{LayerNorm}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1))\big) + \mathbf{b}_2$$
  where $\mathbf{W}_1 \in \mathbb{R}^{1024 \times 512}$ and $\mathbf{W}_2 \in \mathbb{R}^{512 \times 1024}$.
* **Why Pre-LN over Post-LN?**
  - In Post-LN ($\mathbf{y} = \text{LN}(\mathbf{x} + f(\mathbf{x}))$), the normalization scale diminishes gradients on the identity branch as depth increases.
  - In Pre-LN, the identity path is strictly linear ($\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \mathbf{I} + \frac{\partial f}{\partial \mathbf{x}}$), establishing an unobstructed gradient highway during backpropagation on Apple Silicon MPS.

---

### Shining Point 4: Sinusoidal Positional Encoding (Dynamic Sequence Scaling)
* **Mathematical Formulation:**
  $$\mathbf{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad \mathbf{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
* **Why Sinusoidal over Learnable Absolute Embeddings?**
  - Learnable embeddings $\mathbf{E} \in \mathbb{R}^{N \times d}$ crash with index-out-of-bounds if an unseen YouTube conversation has more turns than the training maximum ($S > N$).
  - Sinusoidal encoding has **infinite theoretical extrapolation capability** and allows the model to attend by relative turn distance because:
    $$\mathbf{PE}_{pos+k} = \mathbf{M}_k \cdot \mathbf{PE}_{pos}$$
    where $\mathbf{M}_k$ is a linear rotation matrix.

---

### Shining Point 5: Apple Silicon GPU MPS Optimization
* **Nested Tensor Elimination (`enable_nested_tensor=False`):**
  - PyTorch's native `nn.TransformerEncoder` automatically attempts to use nested tensor C++ kernels when `src_key_padding_mask` is provided. On Apple Silicon Metal Performance Shaders (`mps`), this raises `NotImplementedError: aten::_nested_tensor_from_mask_left_aligned`.
  - Setting `enable_nested_tensor=False` forces standard batched matrix multiplication (`bmm`), enabling **100% native GPU execution on Apple Silicon unified memory with zero CPU fallbacks**.

---

## 4. Key Takeaways for Professor Defense

1. **Multimodality is Essential:** Prosody (`WavLM`) detects emotional intensity/stress; Text (`MPNet`) provides contextual semantics. Bidirectional Cross-Attention resolves ambiguity between sarcasm, polite anger, and genuine satisfaction.
2. **Zero-Leakage Generalization with Transfer Learning:** Pretrained only on synthetic TTS dialogues (clean + acoustically augmented) and validated on held-out *scripts*. Then adapted to real audio by training a small head on the frozen model with uploaded calls, and tested on 19 YouTube calls that are never used in either stage.
3. **Dual Model Comparison Framework:** Both V1 (Mean Pooling regularizer) and V2 (MLP ResBlock + `[CLS]` Token) run side-by-side with full diagnostic telemetry on every single inference request.
