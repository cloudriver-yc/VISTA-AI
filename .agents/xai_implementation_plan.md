# Implementation Plan - Explainable AI (XAI) Layer for Multimodal CSAT Prediction

This plan details the design and implementation of an **Explainability & Attribution Layer (XAI)** for VISTA-AI. It enables the model to explain **WHY** an audio dialogue was classified into a specific category, identifying pivotal dialogue turns, linguistic vs. acoustic dominance, and generating human-readable audit rationales.

---

## User Review Required

> [!IMPORTANT]
> **Zero Breaking Changes to Existing Weights:**  
> All model weights (`models/dual_transformer_v1_weights.pt`, `models/dual_transformer_v2_weights.pt`) will remain **100% compatible**. The forward pass will support an optional `return_xai=True` flag (defaulting to returning logits or a rich XAI dictionary), ensuring existing training routines and evaluation scripts continue to function without retraining.

> [!NOTE]
> **Solving the "Polite Failure" Ambiguity:**  
> For calls like `1735404531.458927.mp3` (where the customer is polite but the withdrawal fails), the XAI layer will explicitly flag the tension discrepancy: showing that while acoustic tone was calm (low prosodic energy), the model's attention concentrated on the failure turning point.

---

## Proposed Changes

### 1. Neural Architecture Core (`src/models/`)

#### [MODIFY] [`src/models/architecture.py`](file:///Users/lorenzolou/Antigravity/tmp/VISTA-AI/src/models/architecture.py)
* **`CrossModalAttention`**:
  * Capture attention weights from `self.text_cross_attn(..., need_weights=True)` and `self.audio_cross_attn(..., need_weights=True)`.
  * Compute the modality energy norms:
    $$\text{Text Norm} = \|\mathbf{t}_{\text{out}}\|_2, \quad \text{Audio Norm} = \|\mathbf{a}_{\text{out}}\|_2$$
    $$\text{Text Ratio} = \frac{\|\mathbf{t}_{\text{out}}\|_2}{\|\mathbf{t}_{\text{out}}\|_2 + \|\mathbf{a}_{\text{out}}\|_2}, \quad \text{Audio Ratio} = 1.0 - \text{Text Ratio}$$
* **`DualTransformerClassifier` (V1) & `EnhancedDualTransformerClassifier` (V2)**:
  * Add parameter `return_xai: bool = False` to `forward()`.
  * Extract turn-level sequence attention saliency $\boldsymbol{\alpha} \in \mathbb{R}^{B \times S}$:
    - For **V1 (Mean Pooling)**: Compute gradient-weighted or normalized projection relevance across the $S$ turns.
    - For **V2 (`[CLS]` Token)**: Extract the `[CLS]` token's cross-turn self-attention weights from the final Transformer Encoder layer ($attn[:, 0, 1:]$).
  * When `return_xai=True`, return:
    ```python
    {
        "logits": logits,                             # (B, 4)
        "probabilities": probs,                       # (B, 4)
        "turn_saliency": turn_saliency,               # (B, S) - attention distribution across turns
        "modality_attribution": {
            "text_ratio": float,                      # e.g., 0.76 (76% text driven)
            "audio_ratio": float                      # e.g., 0.24 (24% acoustic driven)
        },
        "acoustic_norms": audio_norms,                # (B, S) - turn-by-turn vocal tension
        "text_norms": text_norms                      # (B, S) - turn-by-turn semantic magnitude
    }
    ```

---

### 2. Decision Rationale & Explanation Engine (`src/utils/` or `src/models/`)

#### [NEW] [`src/models/explainer.py`](file:///Users/lorenzolou/Antigravity/tmp/VISTA-AI/src/models/explainer.py)
* Encapsulates the interpretation logic:
  * Identifies the **Top-$k$ Pivotal Dialogue Turns** ranked by attention saliency $\alpha_s$.
  * Generates an automated **Natural Language Decision Audit**:
    * Highlights whether the call was dominated by **Linguistic Semantics** (e.g. contract cancellation/error keywords) or **Acoustic Prosody** (e.g. shouting/stress peaks).
    * Evaluates whether the ending was a **Polite Failure** (customer said "thank you" but issue was unresolved) or a **Genuine Resolution**.

---

### 3. User Interfaces & CLI Tooling

#### [MODIFY] [`scripts/05_test_youtube.py`](file:///Users/lorenzolou/Antigravity/tmp/VISTA-AI/scripts/05_test_youtube.py)
* Add an **Explainable AI (XAI) Decision Breakdown** table in terminal output:
  * Modality Ratio gauge: `Text Influence: 74.2% | Audio Influence: 25.8%`.
  * Top-3 Pivotal Turns table with timestamps, speaker, attention weight %, and transcript text.
  * Natural language diagnostic explanation.

#### [MODIFY] [`app.py`](file:///Users/lorenzolou/Antigravity/tmp/VISTA-AI/app.py)
* Add a new expandable card in Streamlit: **"🔍 Explainable AI (XAI): Why did the model make this decision?"**:
  * **Modality Contribution Bar:** Interactive gauge showing text vs. audio balance.
  * **Pivotal Turning Points Table:** Top dialogue turns sorted by decision influence with color-coded saliency badges.
  * **Turn-by-Turn Acoustic vs. Saliency Timeline:** Visual chart displaying vocal tension alongside attention weights over time.
  * **Executive Audit Paragraph:** Plain-language summary explaining the decision drivers.

---

## Verification Plan

### Automated & CLI Tests
1. **Model Forward & XAI Dictionary Output Test:**
   ```bash
   python -c "import torch; from src.models.architecture import EnhancedDualTransformerClassifier; m = EnhancedDualTransformerClassifier(); out = m(torch.randn(1, 10, 768), torch.randn(1, 10, 768), return_xai=True); print('Keys:', out.keys()); print('Saliency shape:', out['turn_saliency'].shape)"
   ```
2. **Backward Compatibility Check:**
   Verify `src/train.py` still runs standard forward passes without errors:
   ```bash
   python src/train.py --model_version both --epochs 1
   ```
3. **CLI Inference with XAI Diagnostic:**
   ```bash
   python scripts/05_test_youtube.py
   ```
   *Verify Top-3 Saliency turns, modality ratio, and text audit display cleanly.*

### Real-World Audit Verification on Uploaded Calls
* Run XAI evaluation on `data/audio/uploads/1735404531.458927.mp3`:
  * Verify that the XAI layer highlights Turn 120–124 (*"Not available in your country"*) as high saliency, explicitly explaining why the failure overrides the polite sign-off.
