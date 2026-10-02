"""
Checks that every GPU-capable ML library in the project runs on the Apple Silicon GPU (MPS).

Each check runs a real workload on MPS and on CPU, compares the outputs and reports the speedup.
CPU-only libraries (scikit-learn, librosa, numba) are listed but not tested; they have no MPS backend.

    python scripts/check_gpu.py            # all checks (downloads HF models / Whisper small.en on first run)
    python scripts/check_gpu.py --quick    # torch + torchvision + project models only, no model downloads
"""
import os
import sys
import time
import argparse
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

warnings.filterwarnings("ignore")
MPS = torch.device("mps")
CPU = torch.device("cpu")
RESULTS = []


def timed(fn, device, repeats=3):
    """Runs fn once to warm up, then returns (output, best wall time in ms) with device sync."""
    out = fn()
    best = float("inf")
    for _ in range(repeats):
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if device.type == "mps":
            torch.mps.synchronize()
        best = min(best, time.perf_counter() - t0)
    return out, best * 1000


def check(name):
    def decorator(fn):
        def run():
            print(f"\n▶ {name}")
            try:
                detail = fn()
                RESULTS.append((name, True, detail))
                print(f"  ✅ {detail}")
            except Exception as e:
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"  ❌ {type(e).__name__}: {e}")
        return run
    return decorator


def compare(name, run_on, rtol=1e-2, atol=1e-3):
    """run_on(device) -> tensor. Returns a summary string after checking MPS output ≈ CPU output."""
    gpu_out, gpu_ms = timed(lambda: run_on(MPS), MPS)
    cpu_out, cpu_ms = timed(lambda: run_on(CPU), CPU)
    gpu_out, cpu_out = gpu_out.float().cpu(), cpu_out.float().cpu()
    max_diff = (gpu_out - cpu_out).abs().max().item()
    if not torch.allclose(gpu_out, cpu_out, rtol=rtol, atol=atol):
        raise AssertionError(f"{name}: MPS and CPU outputs differ (max abs diff {max_diff:.2e})")
    return f"MPS {gpu_ms:.1f} ms vs CPU {cpu_ms:.1f} ms ({cpu_ms / gpu_ms:.1f}x), max diff {max_diff:.1e}"


@check("PyTorch: MPS backend")
def check_torch():
    if not torch.backends.mps.is_built():
        raise RuntimeError("this torch build has no MPS support (x86 Python or a CPU-only wheel?)")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is built but not available (needs macOS 12.3+ on Apple Silicon)")
    a, b = torch.randn(2048, 2048), torch.randn(2048, 2048)
    summary = compare("matmul", lambda d: a.to(d) @ b.to(d), rtol=1e-3, atol=1e-2)
    return f"torch {torch.__version__}, 2048² matmul: {summary}"


@check("PyTorch: autograd / training step on MPS")
def check_training():
    model = torch.nn.Sequential(torch.nn.Linear(768, 256), torch.nn.ReLU(), torch.nn.Linear(256, 4)).to(MPS)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x, y = torch.randn(64, 768, device=MPS), torch.randint(0, 4, (64,), device=MPS)
    losses = []
    for _ in range(20):
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    if not losses[-1] < losses[0]:
        raise AssertionError(f"loss did not decrease ({losses[0]:.3f} -> {losses[-1]:.3f})")
    return f"20 AdamW steps, loss {losses[0]:.3f} -> {losses[-1]:.3f}"


@check("torchvision: ops on MPS")
def check_torchvision():
    import torchvision
    from torchvision.ops import nms
    boxes = torch.rand(500, 4) * 100
    boxes[:, 2:] += boxes[:, :2]
    scores = torch.rand(500)
    keep_gpu = nms(boxes.to(MPS), scores.to(MPS), 0.5).cpu()
    keep_cpu = nms(boxes, scores, 0.5)
    if not torch.equal(keep_gpu, keep_cpu):
        raise AssertionError("NMS results differ between MPS and CPU")
    conv = torchvision.models.resnet18(weights=None).eval()
    x = torch.randn(8, 3, 224, 224)
    with torch.no_grad():
        summary = compare("resnet18", lambda d: conv.to(d)(x.to(d)), rtol=1e-2, atol=1e-2)
    return f"torchvision {torchvision.__version__}, NMS matches; resnet18 batch 8: {summary}"


@check("VISTA models: V1 / V2 classifiers + stage-2 adapters on MPS")
def check_project_models():
    from models.architecture import (DualTransformerClassifier, EnhancedDualTransformerClassifier,
                                     ResidualAdapterHead, AdaptedCSATModel)
    feature = next((os.path.join(paths.FEATURES_DIR, f) for f in sorted(os.listdir(paths.FEATURES_DIR))
                    if f.endswith(".pt")), None) if os.path.isdir(paths.FEATURES_DIR) else None
    if feature:
        data = torch.load(feature, map_location="cpu")
        text, audio, source = data["text_embeds"][None], data["audio_embeds"][None], os.path.basename(feature)
    else:
        text, audio, source = torch.randn(1, 12, 768), torch.randn(1, 12, 768), "random input"

    notes = []
    for tag, cls in (("v1", DualTransformerClassifier), ("v2", EnhancedDualTransformerClassifier)):
        model = cls(num_classes=4, audio_dim=768, text_dim=768, d_model=256, nhead=8, num_layers=2, dropout=0.3)
        weights = os.path.join(paths.MODELS_DIR, f"dual_transformer_{tag}_weights.pt")
        adapter_path = os.path.join(paths.MODELS_DIR, f"csat_adapter_{tag}.pt")
        loaded = "random init"
        if os.path.exists(weights):
            try:
                model.load_state_dict(torch.load(weights, map_location="cpu"))
                loaded = "trained weights"
            except RuntimeError:
                # Checkpoint shape differs from the current architecture (e.g. trained with another d_model);
                # still test MPS with random weights, but say so.
                print(f"  ⚠️  {weights} does not match the current architecture shape; retrain to refresh it")
                loaded = "random init, checkpoint shape mismatch"
            if loaded == "trained weights" and os.path.exists(adapter_path):
                adapter = ResidualAdapterHead()
                adapter.load_state_dict(torch.load(adapter_path, map_location="cpu"))
                model, loaded = AdaptedCSATModel(model, adapter), "trained weights + adapter"
        model.eval()
        with torch.no_grad():
            summary = compare(tag, lambda d: model.to(d)(text.to(d), audio.to(d))[0], rtol=1e-3, atol=1e-3)
            pred = model.to(MPS)(text.to(MPS), audio.to(MPS))[0].argmax(-1).item()
        notes.append(f"{tag.upper()} ({loaded}) pred={pred}: {summary}")
    return f"input {source}\n     " + "\n     ".join(notes)


@check("transformers: WavLM (microsoft/wavlm-base-plus) on MPS")
def check_wavlm():
    import transformers
    from transformers import AutoFeatureExtractor, WavLMModel
    processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval()
    wave = np.random.default_rng(0).standard_normal(16000 * 5).astype(np.float32) * 0.1
    inputs = processor(wave, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        summary = compare("wavlm", lambda d: model.to(d)(inputs["input_values"].to(d)).last_hidden_state.mean(1),
                          rtol=1e-2, atol=1e-2)
    return f"transformers {transformers.__version__}, 5 s clip -> 768-d: {summary}"


@check("sentence-transformers: MPNet (all-mpnet-base-v2) on MPS")
def check_mpnet():
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    if model.device.type != "mps":
        raise AssertionError(f"SentenceTransformer picked {model.device} by default, expected mps")
    sentences = ["I've been waiting on hold for forty minutes.", "Thanks, that fixed it!"] * 16
    summary = compare("mpnet", lambda d: model.to(d).encode(sentences, convert_to_tensor=True, device=str(d)),
                      rtol=1e-2, atol=1e-3)
    return f"sentence-transformers {sentence_transformers.__version__}, default device mps, 32 sentences: {summary}"


@check("openai-whisper: small.en on MPS")
def check_whisper():
    import whisper
    import librosa
    clip = next((os.path.join(paths.TTS_DIR, f) for f in sorted(os.listdir(paths.TTS_DIR))
                 if f.endswith(".wav")), None) if os.path.isdir(paths.TTS_DIR) else None
    if clip:
        audio, _ = librosa.load(clip, sr=16000, mono=True, duration=30)  # soundfile backend, no ffmpeg needed
    else:
        audio = np.zeros(16000 * 5, dtype=np.float32)
    model = whisper.load_model("small.en", device="cpu")
    timings = {}
    for device in (MPS, CPU):
        model = model.to(device)
        t0 = time.perf_counter()
        result = model.transcribe(audio, fp16=False, language="en")
        timings[device.type] = (time.perf_counter() - t0, result["text"].strip())
    (gpu_s, gpu_text), (cpu_s, cpu_text) = timings["mps"], timings["cpu"]
    if clip and not gpu_text:
        raise AssertionError("empty transcript on MPS")
    return (f"{os.path.basename(clip) if clip else 'silence'}: MPS {gpu_s:.1f} s vs CPU {cpu_s:.1f} s "
            f"({cpu_s / gpu_s:.1f}x), transcripts {'match' if gpu_text == cpu_text else 'differ slightly'}\n"
            f"     MPS text: {gpu_text[:100]!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip checks that download pretrained models")
    args = parser.parse_args()

    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | MPS built={torch.backends.mps.is_built()} "
          f"available={torch.backends.mps.is_available()}")
    checks = [check_torch, check_training, check_torchvision, check_project_models]
    if not args.quick:
        checks += [check_wavlm, check_mpnet, check_whisper]
    for run in checks:
        run()

    print("\n" + "=" * 70)
    for name, ok, _ in RESULTS:
        print(f"{'✅' if ok else '❌'} {name}")
    print("ℹ️  CPU-only by design (no MPS backend): scikit-learn, librosa, numba, numpy")
    sys.exit(0 if all(ok for _, ok, _ in RESULTS) else 1)


if __name__ == "__main__":
    main()
