"""
Two-stage training recipe:
  Stage 1 (pretrain): train V1/V2 on synthetic TTS + augmented data (train/val split).
  Stage 2 (finetune): freeze the stage-1 model, add a ResidualAdapterHead on top and train only
                      the adapter on real uploaded calls (finetune split).
  Stage 3 (test):     evaluate stage-1 and stage-2 models on the YouTube calls (test split).
"""
import os
import copy
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.architecture import (DualTransformerClassifier, EnhancedDualTransformerClassifier,
                                 ResidualAdapterHead, AdaptedCSATModel)
from paths import FEATURES_DIR, MODELS_DIR
from data.dataset import RealCSATDataset, pad_collate, get_or_create_splits

CLASSES = ["Very Unsatisfied", "Unsatisfied", "Satisfied", "Very Satisfied"]

MODEL_SPECS = {
    "v1": {"class": DualTransformerClassifier, "name": "V1 Baseline (Linear + Mean Pooling)",
           "weights": "dual_transformer_v1_weights.pt", "adapter": "csat_adapter_v1.pt"},
    "v2": {"class": EnhancedDualTransformerClassifier, "name": "V2 Upgraded (MLP ResBlock + [CLS] Token)",
           "weights": "dual_transformer_v2_weights.pt", "adapter": "csat_adapter_v2.pt"},
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def build_base(spec, device):
    return spec["class"](
        num_classes=4,
        audio_dim=768,
        text_dim=768,
        d_model=512,
        nhead=8,
        num_layers=2,
        dropout=0.3
    ).to(device)


def load_base(spec, device):
    path = os.path.join(MODELS_DIR, spec["weights"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run stage 1 first: python src/train.py --stage pretrain")
    model = build_base(spec, device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()


def load_adapted(spec, device):
    """Stage-2 model (frozen base + adapter), or None if no adapter has been trained yet."""
    path = os.path.join(MODELS_DIR, spec["adapter"])
    if not os.path.exists(path):
        return None
    adapter = ResidualAdapterHead()
    adapter.load_state_dict(torch.load(path, map_location="cpu"))
    return AdaptedCSATModel(load_base(spec, device), adapter).to(device).eval()


def evaluate_on_files(model, files, device):
    """Returns {file: (true_label, predicted_label)} for single-dialogue inference (no padding)."""
    results = {}
    model.eval()
    with torch.no_grad():
        for f in files:
            data = torch.load(f, map_location="cpu")
            t_emb = data["text_embeds"].unsqueeze(0).to(device)
            a_emb = data["audio_embeds"].unsqueeze(0).to(device)
            lbl = data["label"].item() if isinstance(data["label"], torch.Tensor) else int(data["label"])
            logits, _, _ = model(t_emb, a_emb)
            results[f] = (lbl, torch.argmax(logits, dim=1).item())
    return results


def accuracy(results):
    if not results:
        return 0.0
    return 100.0 * sum(t == p for t, p in results.values()) / len(results)


# ---------------------------------------------------------------------------
# Stage 1: pretrain on synthetic data
# ---------------------------------------------------------------------------
def pretrain(spec, device, epochs=25, batch_size=16, lr=2e-4):
    print("\n" + "=" * 75)
    print(f"🚀 Stage 1 · Pretraining {spec['name']} on synthetic data ({device})")
    print("=" * 75)

    model = build_base(spec, device)
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Train/val: synthetic TTS (+ augmented) only
    train_dataset = RealCSATDataset(features_dir=FEATURES_DIR, split="train")
    val_dataset = RealCSATDataset(features_dir=FEATURES_DIR, split="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    print(f"📊 Train: {len(train_dataset)} synthetic | Val: {len(val_dataset)} synthetic (held-out scripts)")

    best_val_acc, best_epoch, best_state, best_train_acc = -1.0, 0, None, 0.0

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for text_embeds, audio_embeds, padding_mask, labels in train_loader:
            text_embeds, audio_embeds = text_embeds.to(device), audio_embeds.to(device)
            padding_mask, labels = padding_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, _, _ = model(text_embeds, audio_embeds, padding_mask=padding_mask)
            loss = classification_criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        # Validation (held-out synthetic scripts)
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for text_embeds, audio_embeds, padding_mask, labels in val_loader:
                text_embeds, audio_embeds = text_embeds.to(device), audio_embeds.to(device)
                padding_mask, labels = padding_mask.to(device), labels.to(device)

                logits, _, _ = model(text_embeds, audio_embeds, padding_mask=padding_mask)
                loss = classification_criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                val_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total
        current_lr = scheduler.get_last_lr()[0]

        if epoch_val_acc > best_val_acc:
            best_val_acc, best_epoch = epoch_val_acc, epoch + 1
            best_train_acc = epoch_train_acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:5.1f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:5.1f}% | LR: {current_lr:.6f}")

    # Keep the checkpoint with the best synthetic validation accuracy
    model.load_state_dict(best_state)
    print(f"\n⭐ Best synthetic val accuracy {best_val_acc:.1f}% at epoch {best_epoch}; using that checkpoint.")

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, spec["weights"])
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved stage-1 weights to {save_path}")

    return {"train_acc": best_train_acc, "val_acc": best_val_acc}


# ---------------------------------------------------------------------------
# Stage 2: freeze the stage-1 model, train only the adapter on uploaded real calls
# ---------------------------------------------------------------------------
def summarize_files(base, files, device):
    """Frozen-base 512-d summaries, base logits and labels for each file (single-dialogue, no padding)."""
    summaries, logits, labels = [], [], []
    with torch.no_grad():
        for f in files:
            data = torch.load(f, map_location="cpu")
            summary = base.encode(data["text_embeds"].unsqueeze(0).to(device), data["audio_embeds"].unsqueeze(0).to(device))
            summaries.append(summary)
            logits.append(base.classifier(summary))
            labels.append(int(data["label"]))
    return torch.cat(summaries), torch.cat(logits), torch.tensor(labels, device=device)


def finetune(spec, device, epochs=30, lr=1e-3, weight_decay=1e-2, seed=42,
             anchor_weight=1.0, anchor_size=512, anchor_batch=64, max_val_drop=1.0):
    """
    Trains only the adapter on uploaded calls. With a handful of uploads the adapter can separate them
    along almost any feature direction and then shift every other call the same way (it collapsed V2
    to predicting 'Unsatisfied'). Two safeguards, neither of which looks at the YouTube test set:
      - anchor: a KL penalty keeps the adapted predictions close to the frozen model's on synthetic
        training calls, so the adapter only makes corrections the uploads really call for;
      - guard: if the adapter lowers synthetic-val accuracy by more than max_val_drop points, it is
        rejected and stage 2 falls back to the stage-1 model.
    """
    print("\n" + "=" * 75)
    print(f"🧩 Stage 2 · Frozen {spec['name']} + adapter, trained on uploaded real calls")
    print("=" * 75)

    splits = get_or_create_splits(features_dir=FEATURES_DIR)
    ft_files = splits["finetune"]
    save_path = os.path.join(MODELS_DIR, spec["adapter"])
    if not ft_files:
        print("⚠️ No uploaded calls (upload_*.pt) in the finetune split; skipping stage 2.")
        return None

    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = AdaptedCSATModel(load_base(spec, device), ResidualAdapterHead()).to(device)
    base_snapshot = {k: v.detach().clone() for k, v in model.base.state_dict().items()}

    # The base is frozen, so its summaries and logits are computed once; training then only runs the adapter
    anchor_files = rng.sample(splits["train"], min(anchor_size, len(splits["train"])))
    up_sum, up_logits, up_labels = summarize_files(model.base, ft_files, device)
    an_sum, an_logits, _ = summarize_files(model.base, anchor_files, device)
    val_sum, val_logits, val_labels = summarize_files(model.base, splits["val"], device)

    counts = torch.bincount(up_labels.cpu(), minlength=4).float()
    print(f"📊 Finetune: {len(ft_files)} uploaded calls | per class: "
          + ", ".join(f"{CLASSES[i]}={int(c)}" for i, c in enumerate(counts))
          + f" | anchor: {len(anchor_files)} synthetic calls (weight {anchor_weight})")
    # Inverse-frequency weights so each class present counts equally
    class_weights = torch.where(counts > 0, counts.sum() / (counts.clamp(min=1) * (counts > 0).sum()), torch.zeros_like(counts))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.adapter.parameters(), lr=lr, weight_decay=weight_decay)
    an_probs = F.softmax(an_logits, dim=-1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = up_logits + model.adapter(up_sum)
        ce = criterion(logits, up_labels)
        idx = torch.randperm(len(an_sum), device=an_sum.device)[:anchor_batch]
        an_adapted = an_logits[idx] + model.adapter(an_sum[idx])
        kl = F.kl_div(F.log_softmax(an_adapted, dim=-1), an_probs[idx], reduction="batchmean")
        loss = ce + anchor_weight * kl
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            acc = 100.0 * (logits.argmax(dim=1) == up_labels).float().mean().item()
            print(f"Adapter epoch [{epoch+1:02d}/{epochs:02d}] | CE: {ce.item():.4f} | Anchor KL: {kl.item():.4f} | Finetune Acc: {acc:5.1f}%")

    for k, v in model.base.state_dict().items():
        assert torch.equal(v, base_snapshot[k]), f"Frozen base parameter changed during stage 2: {k}"
    print("🔒 Verified: stage-1 weights unchanged (only the adapter was trained).")

    # Guard on synthetic val (never on YouTube, which must stay an untouched test set)
    model.eval()
    with torch.no_grad():
        val_base = 100.0 * (val_logits.argmax(dim=1) == val_labels).float().mean().item()
        val_adapted = 100.0 * ((val_logits + model.adapter(val_sum)).argmax(dim=1) == val_labels).float().mean().item()
    print(f"🛡️ Synthetic val · stage 1: {val_base:.1f}% | with adapter: {val_adapted:.1f}% (allowed drop {max_val_drop:.1f} pts)")
    if val_base - val_adapted > max_val_drop:
        if os.path.exists(save_path):
            os.remove(save_path)
        print(f"⛔ Adapter rejected: it lowers synthetic-val accuracy by {val_base - val_adapted:.1f} pts, a sign it "
              "overfit the uploads. Stage 2 falls back to the stage-1 model. Add more uploads, or raise --ft_anchor_weight.")
        return {"finetune_calls": len(ft_files), "accepted": False}

    torch.save(model.adapter.state_dict(), save_path)
    print(f"✅ Saved stage-2 adapter to {save_path}")
    return {"finetune_calls": len(ft_files), "accepted": True}


# ---------------------------------------------------------------------------
# Stage 3: test on YouTube calls
# ---------------------------------------------------------------------------
def test_on_youtube(spec, device):
    print("\n" + "=" * 75)
    print(f"🔍 Stage 3 · {spec['name']} on held-out YouTube calls")
    print("=" * 75)
    test_files = get_or_create_splits(features_dir=FEATURES_DIR)["test"]
    stage1 = evaluate_on_files(load_base(spec, device), test_files, device)
    adapted = load_adapted(spec, device)
    stage2 = evaluate_on_files(adapted, test_files, device) if adapted else {}

    for f in test_files:
        true, p1 = stage1[f]
        line = f"  {'✅' if p1 == true else '❌'} [{os.path.basename(f)}] True: {CLASSES[true]:<16} | Stage 1: {CLASSES[p1]:<16}"
        if stage2:
            p2 = stage2[f][1]
            line += f" | {'✅' if p2 == true else '❌'} Stage 2: {CLASSES[p2]:<16}"
        print(line)

    acc1 = accuracy(stage1)
    if stage2:
        acc2 = accuracy(stage2)
        print(f"🎯 YouTube accuracy · stage 1: {acc1:.1f}% ({len(test_files)} calls) | stage 2: {acc2:.1f}%")
    else:
        # No adapter (not trained, or rejected by the synthetic-val guard): stage 2 serves the stage-1 model,
        # which is also what app.py loads in that case
        acc2 = acc1
        print(f"🎯 YouTube accuracy · stage 1: {acc1:.1f}% ({len(test_files)} calls) | stage 2: no adapter, uses stage 1")
    return {"yt_stage1": acc1, "yt_stage2": acc2, "stage2_fallback": not stage2}


def main():
    parser = argparse.ArgumentParser(description="Train VISTA-AI CSAT Models (two-stage recipe)")
    parser.add_argument("--stage", choices=["pretrain", "finetune", "all"], default="all",
                        help="pretrain = stage 1 on synthetic data; finetune = stage 2 adapter on uploads; all = both. "
                             "The YouTube test (stage 3) always runs at the end.")
    parser.add_argument("--model_version", choices=["v1", "v2", "both"], default="both", help="Model version to train")
    parser.add_argument("--epochs", type=int, default=25, help="Stage-1 training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Stage-1 batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Stage-1 learning rate")
    parser.add_argument("--ft_epochs", type=int, default=30, help="Stage-2 adapter epochs")
    parser.add_argument("--ft_lr", type=float, default=1e-3, help="Stage-2 adapter learning rate")
    parser.add_argument("--ft_anchor_weight", type=float, default=1.0,
                        help="Stage-2 KL penalty keeping adapted predictions close to the frozen model's on synthetic calls")
    parser.add_argument("--ft_max_val_drop", type=float, default=1.0,
                        help="Reject the stage-2 adapter if it lowers synthetic-val accuracy by more than this many points")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the stage-2 adapter initialisation and anchor sample")
    parser.add_argument("--rebuild_split", action="store_true",
                        help="Regenerate data/split_manifest.json from the current feature files before training")
    args = parser.parse_args()

    if args.rebuild_split:
        splits = get_or_create_splits(features_dir=FEATURES_DIR, rebuild=True)
        print(f"🔀 Rebuilt split manifest: {len(splits['train'])} train / {len(splits['val'])} val (synthetic) | "
              f"{len(splits['finetune'])} finetune (uploads) | {len(splits['test'])} test (YouTube)")

    device = get_device()
    versions = ["v1", "v2"] if args.model_version == "both" else [args.model_version]
    results = {}

    for version in versions:
        spec = MODEL_SPECS[version]
        r = {}
        if args.stage in ("pretrain", "all"):
            r.update(pretrain(spec, device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr))
            if version == "v2":
                # Legacy fallback names still used by older loaders
                v2_path = os.path.join(MODELS_DIR, spec["weights"])
                torch.save(torch.load(v2_path), os.path.join(MODELS_DIR, "dual_transformer_weights.pt"))
                torch.save(torch.load(v2_path), os.path.join(MODELS_DIR, "hybrid_cnn_weights.pt"))
            stale_adapter = os.path.join(MODELS_DIR, spec["adapter"])
            if args.stage == "pretrain" and os.path.exists(stale_adapter):
                # An adapter trained on an older stage-1 model no longer matches the new weights
                os.remove(stale_adapter)
                print(f"🗑️ Removed {stale_adapter}: it belonged to the previous stage-1 model. Re-run --stage finetune.")
        if args.stage in ("finetune", "all"):
            finetune(spec, device, epochs=args.ft_epochs, lr=args.ft_lr, seed=args.seed,
                     anchor_weight=args.ft_anchor_weight, max_val_drop=args.ft_max_val_drop)
        r.update(test_on_youtube(spec, device))
        results[spec["name"]] = r

    fmt = lambda v, mark="": f"{v:7.1f}%{mark or ' '}" if v is not None else f"{'-':>9}"
    print("\n" + "=" * 100)
    print("🏆 TWO-STAGE BENCHMARK (stage 1: synthetic pretraining · stage 2: frozen model + adapter on uploads)")
    print("=" * 100)
    print(f"{'Model Architecture':<45} | {'Train':>9} | {'Syn. Val':>9} | {'YT stage1':>9} | {'YT stage2':>9} | {'Δ':>7}")
    print("-" * 100)
    for name, r in results.items():
        delta = r["yt_stage2"] - r["yt_stage1"]
        print(f"{name:<45} | {fmt(r.get('train_acc'))} | {fmt(r.get('val_acc'))} | {fmt(r['yt_stage1'])} | "
              f"{fmt(r['yt_stage2'], '*' if r['stage2_fallback'] else '')} | {delta:+7.1f}")
    print("=" * 100)
    if any(r["stage2_fallback"] for r in results.values()):
        print("* No adapter for this model (not trained, or rejected by the synthetic-val guard): stage 2 uses the stage-1 model.")

if __name__ == "__main__":
    main()
