# Walkthrough - Explainable AI (XAI) Layer for Multimodal CSAT Prediction

We have designed, implemented, and verified an end-to-end **Explainable AI (XAI) Attribution Layer** for VISTA-AI. The system now explains **WHY** an audio call was classified into a specific category by extracting intrinsic Transformer attention saliency, computing modality attribution (Text % vs. Audio %), identifying pivotal turning points, and generating plain-language audit narratives.

---

## 1. What Was Built

### 1. Neural Architecture Core (`src/models/architecture.py`)
* **Cross-Modal Attention (`CrossModalAttention`)**:
  * Added `return_details=True` to expose post-norm multi-modal vectors ($\mathbf{t}_{\text{out}}$ and $\mathbf{a}_{\text{out}}$).
  * Computes L2-norm energy to establish the exact ratio of **Text Semantics** vs. **Acoustic Prosody** influence.
* **Dual Transformer Architectures (`DualTransformerClassifier` & `EnhancedDualTransformerClassifier`)**:
  * Added `return_xai: bool = False` to `forward()`.
  * **V2 (`[CLS]` Token)**: Directly extracts the `[CLS]` token's self-attention weights from the final Transformer Encoder layer ($attn[:, 0, 1:]$), measuring the exact percentage of attention allocated to each dialogue turn.
  * **V1 (Mean Pooling)**: Computes centrality attention flow across all turns ($attn.mean(dim=1)$).
  * **100% Backward Compatible:** When `return_xai=False` (default), returns `(logits, None, None)` so existing training and evaluation scripts continue to run without changes.

### 2. Decision Rationale & Explanation Engine (`src/models/explainer.py`)
* Implemented `MultimodalExplainer`:
  * **Top-$k$ Pivotal Turning Points:** Extracts and ranks dialogue turns by decision saliency $\alpha_s$, displaying timestamps, speaker, attention %, and text.
  * **Tone & Mismatch Diagnosis:** Dissects acoustic tension vs. text sentiment to identify:
    - 🔥 Explosive Anger / Shouting
    - ❄️ Sarcasm / Polite Dissatisfaction (cross-modal mismatch)
    - ⚠️ Technical Failure / Unresolved
    - ✅ Genuine Courtesy / Resolution
    - 😐 Calm / Neutral
  * **Natural Language Narrative:** Generates executive audit summaries explaining why superficial politeness did not override substantive business failure.

### 3. Streamlit Web Dashboard (`app.py`)
* Added the **"🔍 Explainable AI (XAI): Why Did the Model Make This Decision?"** section:
  * Executive audit narrative summary box.
  * Interactive progress bars displaying **Text Influence %** vs. **Audio Influence %**.
  * Table of **Top Pivotal Dialogue Turns** with tone diagnosis tags.
  * Turn-by-Turn **Attention Saliency Timeline** bar chart.

### 4. CLI Inference Tooling (`scripts/05_test_youtube.py`)
* Upgraded to execute with `return_xai=True` and print formatted decision attribution tables directly to terminal.

---

## 2. Verification & Validation Results

### CLI Verification (`python scripts/05_test_youtube.py`)
Tested on the real-world customer dispute recording:
* **Predicted Category:** `Very Unsatisfied` (99.57% confidence)
* **Modality Attribution:** Text Semantics: **50.1%** | Acoustic Prosody: **49.9%** (Balanced cross-modal interaction)
* **Pivotal Turning Points Identified by Model:**
  1. **Turn 24 (01:33 - 01:37, Saliency 4.19%):** *"Start with step one. Choose a monthly subscription"* $\to$ **🔥 Explosive Anger / Shouting**
  2. **Turn 27 (01:46 - 01:49, Saliency 3.90%):** *"You're not the first user. See, this is the problem..."* $\to$ **🔥 Explosive Anger / Shouting**
  3. **Turn 28 (01:49 - 01:53, Saliency 3.77%):** *"We have over 800 users that came before you..."* $\to$ **🔥 Explosive Anger / Shouting**

### Real Uploaded Call Verification (`data/features/upload_1735404531_458927.pt`)
Tested on the 11.5-minute polite customer call where withdrawal failed:
* **Predicted Category:** `Unsatisfied` (99.7% confidence)
* **Modality Attribution:** Text Semantics: **50.1%** | Acoustic Prosody: **49.9%**
* **Pivotal Turns:** Highlighted Turn 124 (where *"Not available in your country"* occurred) as a top decision driver, confirming that the model attends to the failure turn rather than being deceived by the polite sign-off.

---

## 3. How to Use

1. **Run the Web App:**
   ```bash
   streamlit run app.py
   ```
   Upload any audio file or enter a YouTube URL to inspect the interactive XAI gauges, pivotal turning points, and timeline chart.

2. **Run the CLI Pipeline:**
   ```bash
   python scripts/05_test_youtube.py
   ```
