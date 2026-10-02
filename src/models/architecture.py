import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Bidirectional Cross-Modal Attention:
    - Text queries Audio features (acoustic-grounded semantics)
    - Audio queries Text features (semantic-grounded prosody)
    """
    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super().__init__()
        self.text_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.audio_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_a = nn.LayerNorm(d_model)
        
        self.fuse_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, text_proj, audio_proj, return_details=False):
        # text_proj, audio_proj: (Batch, Segments, d_model)
        t_attended, _ = self.text_cross_attn(query=text_proj, key=audio_proj, value=audio_proj)
        t_out = self.norm_t(text_proj + t_attended)
        
        a_attended, _ = self.audio_cross_attn(query=audio_proj, key=text_proj, value=text_proj)
        a_out = self.norm_a(audio_proj + a_attended)
        
        fused = torch.cat([t_out, a_out], dim=-1) # (Batch, Segments, 2 * d_model)
        out = self.fuse_proj(fused) # (Batch, Segments, d_model)
        if return_details:
            return out, {"t_out": t_out, "a_out": a_out}
        return out


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (Batch, Segments, d_model)
        S = x.size(1)
        if S > self.pe.size(1):
            # Extend buffer dynamically if conversation exceeds 1024 chunks
            device = x.device
            max_len = S + 128
            pe = torch.zeros(max_len, x.size(2), device=device)
            position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, x.size(2), 2, device=device).float() * (-math.log(10000.0) / x.size(2)))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return x + pe.unsqueeze(0)[:, :S, :]
        return x + self.pe[:, :S, :]


class MLPResBlock(nn.Module):
    """
    Pre-LN 2-layer MLP Residual Block for feature adaptation:
    h = x + Dropout(Linear2(GELU(LayerNorm(Linear1(x)))))
    Allows non-linear adaptation of pre-trained embeddings while stabilizing gradient flow.
    """
    def __init__(self, d_model=256, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)


class DualTransformerClassifier(nn.Module):
    """
    V1 Baseline Architecture:
    - Linear Projections
    - Bidirectional Cross-Modal Attention Fusion
    - Conversational Sequence Transformer Encoder
    - Mean-Pooled CSAT Classification Head
    """
    def __init__(self, num_classes=4, audio_dim=768, text_dim=768, d_model=256, nhead=8, num_layers=2, max_len=1024, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        
        # Projection from original feature dims to unified d_model
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Cross-Modal Attention
        self.cross_modal_fusion = CrossModalAttention(d_model=d_model, nhead=nhead, dropout=0.1)
        
        # Positional Encoding for conversational turn progression
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
        
        # Conversational Sequence Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.2,
            activation="gelu",
            batch_first=True
        )
        self.sequence_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        
        # Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

    @staticmethod
    def _pool(seq_out, padding_mask):
        if padding_mask is not None:
            mask_expanded = (~padding_mask).unsqueeze(-1).float() # (B, S, 1)
            sum_embeddings = torch.sum(seq_out * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask # (B, d_model)
        return seq_out.mean(dim=1) # (B, d_model)

    def encode(self, text_embeds, audio_embeds, padding_mask=None):
        """Returns the (B, d_model) dialogue summary that feeds the classifier (used by adapter heads)."""
        t_proj = self.text_proj(text_embeds)
        a_proj = self.audio_proj(audio_embeds)
        h = self.pos_encoder(self.cross_modal_fusion(t_proj, a_proj))
        if padding_mask is not None:
            seq_out = self.sequence_transformer(h, src_key_padding_mask=padding_mask)
        else:
            seq_out = self.sequence_transformer(h)
        return self._pool(seq_out, padding_mask)

    def forward(self, text_embeds, audio_embeds, padding_mask=None, return_xai=False):
        """
        Inputs:
        - text_embeds: (Batch, Segments, 768)
        - audio_embeds: (Batch, Segments, 768)
        - padding_mask: (Batch, Segments) - True for padded positions
        - return_xai: (bool) - if True, returns dictionary with attention saliency and modality attribution
        """
        if not return_xai:
            return self.classifier(self.encode(text_embeds, audio_embeds, padding_mask)), None, None

        B, S, _ = text_embeds.size()

        # 1. Project both modalities to unified d_model
        t_proj = self.text_proj(text_embeds)   # (B, S, d_model)
        a_proj = self.audio_proj(audio_embeds) # (B, S, d_model)

        # 2. Cross-Modal Fusion
        fused, fusion_info = self.cross_modal_fusion(t_proj, a_proj, return_details=True)

        # 3. Add positional embeddings (turn progression)
        h = self.pos_encoder(fused)

        # 4. Process conversational dynamics across turns (manual loop to capture attention weights)
        last_attn = None
        curr_h = h
        for layer in self.sequence_transformer.layers:
            attn_out, attn_weights = layer.self_attn(
                curr_h, curr_h, curr_h,
                key_padding_mask=padding_mask,
                need_weights=True,
                average_attn_weights=True
            )
            curr_h = layer.norm1(curr_h + layer.dropout1(attn_out))
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(curr_h))))
            curr_h = layer.norm2(curr_h + layer.dropout2(ff_out))
            last_attn = attn_weights
        seq_out = curr_h

        pooled = self._pool(seq_out, padding_mask)

        # 5. CSAT Logits
        logits = self.classifier(pooled) # (B, num_classes)

        if return_xai:
            probs = F.softmax(logits, dim=-1)
            if last_attn is not None:
                raw_saliency = last_attn.mean(dim=1) # (B, S)
                if padding_mask is not None:
                    raw_saliency = raw_saliency.masked_fill(padding_mask, 0.0)
                turn_saliency = raw_saliency / torch.clamp(raw_saliency.sum(dim=-1, keepdim=True), min=1e-9)
            else:
                turn_saliency = torch.ones((B, S), device=logits.device) / S

            t_out = fusion_info["t_out"] # (B, S, d_model)
            a_out = fusion_info["a_out"] # (B, S, d_model)

            # Compute modality contribution via fuse_proj linear layer weights and activations
            w = self.cross_modal_fusion.fuse_proj[0].weight # (d_model, 2 * d_model)
            w_t = w[:, :self.d_model]
            w_a = w[:, self.d_model:]
            h_t = torch.matmul(t_out, w_t.t()) # (B, S, d_model)
            h_a = torch.matmul(a_out, w_a.t()) # (B, S, d_model)
            norm_ht = torch.norm(h_t, p=2, dim=-1).mean(dim=-1, keepdim=True)
            norm_ha = torch.norm(h_a, p=2, dim=-1).mean(dim=-1, keepdim=True)
            text_ratio = (norm_ht / torch.clamp(norm_ht + norm_ha, min=1e-9)).squeeze(-1)
            audio_ratio = 1.0 - text_ratio

            # Real acoustic tension & text embedding norms from input features
            raw_a_norm = torch.norm(audio_embeds, p=2, dim=-1) # (B, S)
            raw_t_norm = torch.norm(text_embeds, p=2, dim=-1) # (B, S)

            # Cross-modal directional discrepancy between projected modalities
            cos_sim = F.cosine_similarity(t_proj, a_proj, dim=-1) # (B, S)
            cross_mismatch = (1.0 - cos_sim) / 2.0 # [0, 1]

            return {
                "logits": logits,
                "probabilities": probs,
                "predicted_class": logits.argmax(dim=-1).item(),
                "turn_saliency": turn_saliency.squeeze(0),
                "text_ratio": float(text_ratio[0].item()),
                "audio_ratio": float(audio_ratio[0].item()),
                "text_norms": raw_t_norm.squeeze(0),
                "audio_norms": raw_a_norm.squeeze(0),
                "turn_cross_mismatch": cross_mismatch.squeeze(0)
            }

        return logits, None, None


class EnhancedDualTransformerClassifier(nn.Module):
    """
    V2 Upgraded Architecture:
    - 1D Projection + MLPResBlock Adaptation Head
    - Bidirectional Cross-Modal Attention
    - Learnable [CLS] Token Prepending
    - Sinusoidal Positional Encoding
    - Conversational Sequence Transformer Encoder
    - CSAT Classification from [CLS] Token Representation
    """
    def __init__(self, num_classes=4, audio_dim=768, text_dim=768, d_model=256, nhead=8, num_layers=2, max_len=1024, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        
        # 1. Projections with MLP Residual Adaptation
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            MLPResBlock(d_model=d_model, hidden_dim=d_model * 2, dropout=0.1)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            MLPResBlock(d_model=d_model, hidden_dim=d_model * 2, dropout=0.1)
        )
        
        # 2. Cross-Modal Attention
        self.cross_modal_fusion = CrossModalAttention(d_model=d_model, nhead=nhead, dropout=0.1)
        
        # 3. Learnable [CLS] Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # 4. Positional Encoding
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
        
        # 5. Sequence Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.2,
            activation="gelu",
            batch_first=True
        )
        self.sequence_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        
        # 6. Classification Head on [CLS] Token
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

    def _prepend_cls(self, fused, padding_mask):
        """Prepends the [CLS] token (never padded), adds positional encoding, and extends the mask."""
        B = fused.size(0)
        seq_with_cls = torch.cat([self.cls_token.expand(B, -1, -1), fused], dim=1) # (B, 1 + S, d_model)
        h = self.pos_encoder(seq_with_cls)
        if padding_mask is None:
            return h, None
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=padding_mask.device)
        return h, torch.cat([cls_mask, padding_mask], dim=1)

    def encode(self, text_embeds, audio_embeds, padding_mask=None):
        """Returns the (B, d_model) [CLS] representation that feeds the classifier (used by adapter heads)."""
        t_proj = self.text_proj(text_embeds)
        a_proj = self.audio_proj(audio_embeds)
        h, mask_with_cls = self._prepend_cls(self.cross_modal_fusion(t_proj, a_proj), padding_mask)
        if mask_with_cls is not None:
            seq_out = self.sequence_transformer(h, src_key_padding_mask=mask_with_cls)
        else:
            seq_out = self.sequence_transformer(h)
        return seq_out[:, 0, :]

    def forward(self, text_embeds, audio_embeds, padding_mask=None, return_xai=False):
        """
        Inputs:
        - text_embeds: (Batch, Segments, 768)
        - audio_embeds: (Batch, Segments, 768)
        - padding_mask: (Batch, Segments) - True for padded positions
        - return_xai: (bool) - if True, returns dictionary with attention saliency and modality attribution
        """
        if not return_xai:
            return self.classifier(self.encode(text_embeds, audio_embeds, padding_mask)), None, None

        B, S, _ = text_embeds.size()

        # 1. Project both modalities through MLP ResBlocks
        t_proj = self.text_proj(text_embeds)   # (B, S, d_model)
        a_proj = self.audio_proj(audio_embeds) # (B, S, d_model)

        # 2. Cross-Modal Fusion
        fused, fusion_info = self.cross_modal_fusion(t_proj, a_proj, return_details=True)

        # 3-5. Prepend [CLS], add positional encoding, extend padding mask (CLS is never padded)
        h, mask_with_cls = self._prepend_cls(fused, padding_mask)

        # Manual encoder loop to capture attention weights
        last_attn = None
        curr_h = h
        for layer in self.sequence_transformer.layers:
            attn_out, attn_weights = layer.self_attn(
                curr_h, curr_h, curr_h,
                key_padding_mask=mask_with_cls,
                need_weights=True,
                average_attn_weights=True
            )
            curr_h = layer.norm1(curr_h + layer.dropout1(attn_out))
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(curr_h))))
            curr_h = layer.norm2(curr_h + layer.dropout2(ff_out))
            last_attn = attn_weights
        seq_out = curr_h

        # 6. Extract [CLS] token representation (index 0)
        cls_rep = seq_out[:, 0, :] # (B, d_model)
        
        # 7. CSAT Classification
        logits = self.classifier(cls_rep)

        if return_xai:
            probs = F.softmax(logits, dim=-1)
            # CLS token is at index 0, so its attention over the turns is last_attn[:, 0, 1:]
            if last_attn is not None:
                raw_saliency = last_attn[:, 0, 1:] # (B, S)
                if padding_mask is not None:
                    raw_saliency = raw_saliency.masked_fill(padding_mask, 0.0)
                turn_saliency = raw_saliency / torch.clamp(raw_saliency.sum(dim=-1, keepdim=True), min=1e-9)
            else:
                turn_saliency = torch.ones((B, S), device=logits.device) / S

            t_out = fusion_info["t_out"] # (B, S, d_model)
            a_out = fusion_info["a_out"] # (B, S, d_model)

            # Compute modality contribution via fuse_proj linear layer weights and activations
            w = self.cross_modal_fusion.fuse_proj[0].weight # (d_model, 2 * d_model)
            w_t = w[:, :self.d_model]
            w_a = w[:, self.d_model:]
            h_t = torch.matmul(t_out, w_t.t()) # (B, S, d_model)
            h_a = torch.matmul(a_out, w_a.t()) # (B, S, d_model)
            norm_ht = torch.norm(h_t, p=2, dim=-1).mean(dim=-1, keepdim=True)
            norm_ha = torch.norm(h_a, p=2, dim=-1).mean(dim=-1, keepdim=True)
            text_ratio = (norm_ht / torch.clamp(norm_ht + norm_ha, min=1e-9)).squeeze(-1)
            audio_ratio = 1.0 - text_ratio

            # Real acoustic tension & text embedding norms from input features
            raw_a_norm = torch.norm(audio_embeds, p=2, dim=-1) # (B, S)
            raw_t_norm = torch.norm(text_embeds, p=2, dim=-1) # (B, S)

            # Cross-modal directional discrepancy between projected modalities
            cos_sim = F.cosine_similarity(t_proj, a_proj, dim=-1) # (B, S)
            cross_mismatch = (1.0 - cos_sim) / 2.0 # [0, 1]

            return {
                "logits": logits,
                "probabilities": probs,
                "predicted_class": logits.argmax(dim=-1).item(),
                "turn_saliency": turn_saliency.squeeze(0),
                "text_ratio": float(text_ratio[0].item()),
                "audio_ratio": float(audio_ratio[0].item()),
                "text_norms": raw_t_norm.squeeze(0),
                "audio_norms": raw_a_norm.squeeze(0),
                "turn_cross_mismatch": cross_mismatch.squeeze(0)
            }

        return logits, None, None


class ResidualAdapterHead(nn.Module):
    """
    Small head trained on real calls on top of a frozen, synthetically pretrained model.
    Its output is added to the frozen model's logits. The last layer is zero-initialised,
    so before any training the adapted model predicts exactly what the base model predicts.
    """
    def __init__(self, d_model=256, hidden_dim=64, num_classes=4, dropout=0.3):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.out = nn.Linear(hidden_dim, num_classes)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, summary):
        return self.out(self.hidden(summary))


class AdaptedCSATModel(nn.Module):
    """
    Stage-2 model: a frozen V1/V2 base plus a trainable ResidualAdapterHead.
    Same call signature and return values as the base models, so training, the CLI and the app
    can use it in place of a base model. XAI fields (saliency, modality ratio, ...) come from the
    frozen base; only the logits/probabilities/prediction are replaced by the adapted ones.
    """
    def __init__(self, base, adapter):
        super().__init__()
        self.base = base
        self.adapter = adapter
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # the frozen base never runs dropout
        return self

    def adapted_logits(self, text_embeds, audio_embeds, padding_mask=None):
        summary = self.base.encode(text_embeds, audio_embeds, padding_mask)
        return self.base.classifier(summary) + self.adapter(summary)

    def forward(self, text_embeds, audio_embeds, padding_mask=None, return_xai=False):
        logits = self.adapted_logits(text_embeds, audio_embeds, padding_mask)
        if not return_xai:
            return logits, None, None
        xai = self.base(text_embeds, audio_embeds, padding_mask=padding_mask, return_xai=True)
        xai["logits"] = logits
        xai["probabilities"] = F.softmax(logits, dim=-1)
        xai["predicted_class"] = logits.argmax(dim=-1).item()
        return xai


def with_adapter_if_available(base, adapter_path, device):
    """Wraps a loaded stage-1 model with its stage-2 adapter when the adapter file exists; otherwise returns it unchanged."""
    import os
    if not os.path.exists(adapter_path):
        return base
    adapter = ResidualAdapterHead()
    adapter.load_state_dict(torch.load(adapter_path, map_location="cpu"))
    return AdaptedCSATModel(base, adapter).to(device).eval()


class YShapedHybridCNN(nn.Module):
    """
    Legacy Hybrid Architecture:
    - Audio: 2D CNN over Log-Mel Spectrograms
    - Text: Pre-trained Transformer embeddings (768-dim) directly inputted
    """
    def __init__(self, num_classes=4, text_dim=768):
        super().__init__()
        
        # Audio 2D CNN Branch (Expects input shape: B, 1, 128, T)
        self.audio_branch = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten() # Output: (B, 128)
        )
        
        # Shared Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(128 + text_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, text_embeds, mel_spec):
        """
        Inputs:
        - text_embeds: (Batch, Segments, 768)
        - mel_spec: (Batch, Segments, 1, 128, 312)
        """
        B, S, C, H, W = mel_spec.size()
        
        # Flatten Batch and Segments for parallel processing
        mel_spec_flat = mel_spec.view(B * S, C, H, W)
        text_embeds_flat = text_embeds.view(B * S, -1)
        
        h_audio_flat = self.audio_branch(mel_spec_flat) # (B*S, 128)
        
        # 1D Vector Concatenation per segment
        fused_flat = torch.cat([h_audio_flat, text_embeds_flat], dim=1) # (B*S, 896)
        
        # Shared Classification Head per segment
        logits_flat = self.classifier(fused_flat) # (B*S, num_classes)
        
        # Reshape back to sequence
        logits_seq = logits_flat.view(B, S, -1) # (B, S, num_classes)
        
        # Late Aggregation: Mean Pooling across segments
        logits = logits_seq.mean(dim=1) # (B, num_classes)
        
        return logits, None, None
 # Return None for contrastive projections to match API
