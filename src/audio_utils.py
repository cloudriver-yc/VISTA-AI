import numpy as np

# WavLM's conv feature encoder has a 400-sample (25 ms @ 16 kHz) receptive field; shorter input crashes it
MIN_WAVLM_SAMPLES = 400


def segment_audio(audio, start_sec, end_sec, sr, min_samples=MIN_WAVLM_SAMPLES):
    """Slice a Whisper segment, widening very short ones around their centre to WavLM's minimum length.
    Returns None when the timestamps fall outside the audio (Whisper occasionally overshoots the end)."""
    start = max(0, int(start_sec * sr))
    end = min(len(audio), int(end_sec * sr))
    if start >= len(audio) or end <= start:
        return None
    if end - start < min_samples:
        centre = (start + end) // 2
        start = max(0, centre - min_samples // 2)
        end = min(len(audio), start + min_samples)
        start = max(0, end - min_samples)
    seg = audio[start:end]
    if len(seg) < min_samples:  # whole clip shorter than 25 ms
        seg = np.pad(seg, (0, min_samples - len(seg)))
    return seg
