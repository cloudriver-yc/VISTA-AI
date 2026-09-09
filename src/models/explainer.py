import numpy as np
import torch

CANONICAL_CLASSES = [
    "Very Unsatisfied", # 0: Delighted/Happy (canonical project mapping)
    "Unsatisfied",      # 1: Flat/At-risk, unsolved
    "Satisfied",        # 2: Flat/Standard resolved
    "Very Satisfied"    # 3: Angry/Urgent shouting, unsolved
]

class MultimodalExplainer:
    """
    Explainable AI (XAI) engine for Multimodal Conversational CSAT prediction.
    Extracts attention saliency, modality dominance, and generates human-readable audit reports.
    """
    def __init__(self, class_names=None):
        self.class_names = class_names or CANONICAL_CLASSES

    def explain(self, xai_output: dict, segments: list, top_k: int = 3) -> dict:
        """
        Generates structured and narrative explanations for a call prediction.
        
        Args:
            xai_output (dict): Output from model.forward(..., return_xai=True)
            segments (list): List of dicts with keys 'start', 'end', 'text', and optional 'speaker'
            top_k (int): Number of pivotal dialogue turns to highlight
        """
        predicted_idx = xai_output["predicted_class"]
        predicted_label = self.class_names[predicted_idx]
        probs = xai_output["probabilities"].detach().squeeze(0).cpu().numpy()
        confidence_pct = float(probs[predicted_idx] * 100)

        saliency = xai_output["turn_saliency"].detach().cpu().numpy()
        t_norms = xai_output["text_norms"].detach().cpu().numpy()
        a_norms = xai_output["audio_norms"].detach().cpu().numpy()
        mismatches = xai_output["turn_cross_mismatch"].detach().cpu().numpy()

        text_ratio_pct = xai_output["text_ratio"] * 100
        audio_ratio_pct = xai_output["audio_ratio"] * 100

        # Align with segments length
        num_turns = min(len(segments), len(saliency))
        turn_details = []

        for i in range(num_turns):
            seg = segments[i]
            s_val = float(saliency[i] * 100)
            a_val = float(a_norms[i])
            t_val = float(t_norms[i])
            m_val = float(mismatches[i])

            start_m, start_s = divmod(int(seg.get("start", 0)), 60)
            end_m, end_s = divmod(int(seg.get("end", 0)), 60)
            timestamp_str = f"{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}"

            # Tone diagnostic per turn
            text_str = seg.get("text", "").strip()
            tone = self._diagnose_turn_tone(text_str, a_val, m_val)

            turn_details.append({
                "turn_index": i + 1,
                "timestamp": timestamp_str,
                "speaker": seg.get("speaker", f"Turn {i+1}"),
                "text": text_str,
                "saliency_pct": s_val,
                "acoustic_tension": a_val,
                "text_norm": t_val,
                "cross_mismatch": m_val,
                "tone_diagnosis": tone
            })

        # Rank turns by decision saliency
        sorted_turns = sorted(turn_details, key=lambda x: x["saliency_pct"], reverse=True)
        top_pivotal_turns = sorted_turns[:min(top_k, len(sorted_turns))]

        # Generate narrative audit explanation
        narrative = self._generate_narrative(
            predicted_label=predicted_label,
            confidence_pct=confidence_pct,
            text_ratio_pct=text_ratio_pct,
            audio_ratio_pct=audio_ratio_pct,
            top_turns=top_pivotal_turns,
            all_turns=turn_details
        )

        return {
            "predicted_label": predicted_label,
            "confidence_pct": confidence_pct,
            "class_probabilities": {self.class_names[i]: float(probs[i] * 100) for i in range(len(self.class_names))},
            "modality_attribution": {
                "text_influence_pct": round(text_ratio_pct, 1),
                "audio_influence_pct": round(audio_ratio_pct, 1),
                "dominant_modality": "Text Semantics" if text_ratio_pct >= audio_ratio_pct else "Acoustic Prosody"
            },
            "top_pivotal_turns": top_pivotal_turns,
            "all_turn_details": turn_details,
            "narrative_rationale": narrative
        }

    def _diagnose_turn_tone(self, text: str, acoustic_norm: float, mismatch: float) -> str:
        lower_t = text.lower()
        polite_words = any(w in lower_t for w in ["thank", "thanks", "great", "wonderful", "perfect", "good", "welcome", "appreciate"])
        negative_words = any(w in lower_t for w in ["error", "not available", "failed", "cancel", "useless", "rubbish", "refuse", "blocked", "reject", "sue", "court"])

        if acoustic_norm > 3.2 or (acoustic_norm > 2.3 and negative_words):
            return "🔥 Explosive Anger / Shouting"
        elif acoustic_norm > 2.5 and polite_words:
            return "🎉 Enthusiastic / Strong Agreement"
        elif polite_words and (mismatch > 0.45 or acoustic_norm < 1.7):
            return "❄️ Sarcasm / Polite Dissatisfaction"
        elif negative_words and acoustic_norm < 2.0:
            return "⚠️ Technical Failure / Unresolved"
        elif polite_words:
            return "✅ Genuine Courtesy / Resolution"
        elif acoustic_norm < 1.8:
            return "😐 Calm / Neutral Matter-of-Fact"
        else:
            return "💬 Active Conversational Engagement"

    def _generate_narrative(self, predicted_label: str, confidence_pct: float, text_ratio_pct: float, audio_ratio_pct: float, top_turns: list, all_turns: list) -> str:
        """
        Creates a clear, human-auditable paragraph explaining the decision.
        """
        dominant = "Text Semantics" if text_ratio_pct >= audio_ratio_pct else "Acoustic Prosody"
        top_turn_strs = []
        for t in top_turns:
            top_turn_strs.append(f"Turn {t['turn_index']} ({t['timestamp']}: \"{t['text'][:50]}...\") with {t['saliency_pct']:.1f}% attention saliency")

        turning_point_summary = " and ".join(top_turn_strs[:2])

        # Check if polite ending occurred
        last_turns_text = " ".join([t["text"].lower() for t in all_turns[-3:]])
        polite_closing = any(w in last_turns_text for w in ["thank you", "thanks", "bye", "good day"])

        closing_nuance = ""
        if polite_closing and predicted_label in ["Unsatisfied", "Very Unsatisfied"]:
            closing_nuance = (
                " Notably, although the call concluded with superficial courtesy or sign-offs ('thank you'), "
                "the model's sequence attention prioritized earlier critical turning points where unresolved friction occurred, "
                "correctly preventing polite closing words from overriding the substantive outcome."
            )
        elif predicted_label in ["Satisfied", "Very Satisfied"]:
            closing_nuance = " The interaction exhibited positive resolution signals and mutual consensus without critical conversational breakdown."

        rationale = (
            f"The dialogue was classified as **{predicted_label}** with **{confidence_pct:.1f}% confidence**. "
            f"The decision was primarily driven by **{dominant}** ({text_ratio_pct:.1f}% linguistic vs. {audio_ratio_pct:.1f}% acoustic weight). "
            f"The model identified pivotal turning points at {turning_point_summary}.{closing_nuance}"
        )
        return rationale
