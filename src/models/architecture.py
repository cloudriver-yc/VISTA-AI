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
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
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
    def __init__(self, d_model=512, max_len=1024):
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
    def __init__(self, d_model=512, hidden_dim=1024, dropout=0.1):
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
    def __init__(self, num_classes=4, audio_dim=768, text_dim=768, d_model=512, nhead=8, num_layers=2, max_len=1024, dropout=0.3):
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

    def forward(self, text_embeds, audio_embeds, padding_mask=None, return_xai=False):
        """
        Inputs:
        - text_embeds: (Batch, Segments, 768)
        - audio_embeds: (Batch, Segments, 768)
        - padding_mask: (Batch, Segments) - True for padded positions
        - return_xai: (bool) - if True, returns dictionary with attention saliency and modality attribution
        """
        B, S, _ = text_embeds.size()
        
        # 1. Project both modalities to unified d_model
        t_proj = self.text_proj(text_embeds)   # (B, S, d_model)
        a_proj = self.audio_proj(audio_embeds) # (B, S, d_model)
        
        # 2. Cross-Modal Fusion
        if return_xai:
            fused, fusion_info = self.cross_modal_fusion(t_proj, a_proj, return_details=True)
        else:
            fused = self.cross_modal_fusion(t_proj, a_proj)
        
        # 3. Add positional embeddings (turn progression)
        h = self.pos_encoder(fused)
        
        # 4. Process conversational dynamics across turns
        last_attn = None
        if return_xai:
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
        else:
            if padding_mask is not None:
                seq_out = self.sequence_transformer(h, src_key_padding_mask=padding_mask)
            else:
                seq_out = self.sequence_transformer(h)

        if padding_mask is not None:
            mask_expanded = (~padding_mask).unsqueeze(-1).float() # (B, S, 1)
            sum_embeddings = torch.sum(seq_out * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask # (B, d_model)
        else:
            pooled = seq_out.mean(dim=1) # (B, d_model)
            
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
            t_norm = torch.norm(t_out, p=2, dim=-1) # (B, S)
            a_norm = torch.norm(a_out, p=2, dim=-1) # (B, S)
            mean_t = t_norm.mean(dim=-1, keepdim=True)
            mean_a = a_norm.mean(dim=-1, keepdim=True)
            text_ratio = (mean_t / torch.clamp(mean_t + mean_a, min=1e-9)).squeeze(-1)
            audio_ratio = 1.0 - text_ratio

            return {
                "logits": logits,
                "probabilities": probs,
                "predicted_class": logits.argmax(dim=-1).item(),
                "turn_saliency": turn_saliency.squeeze(0),
                "text_ratio": float(text_ratio[0].item()),
                "audio_ratio": float(audio_ratio[0].item()),
                "text_norms": t_norm.squeeze(0),
                "audio_norms": a_norm.squeeze(0),
                "turn_cross_mismatch": (torch.abs(t_norm - a_norm) / torch.clamp(t_norm + a_norm, min=1e-9)).squeeze(0)
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
    def __init__(self, num_classes=4, audio_dim=768, text_dim=768, d_model=512, nhead=8, num_layers=2, max_len=1024, dropout=0.3):
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

    def forward(self, text_embeds, audio_embeds, padding_mask=None, return_xai=False):
        """
        Inputs:
        - text_embeds: (Batch, Segments, 768)
        - audio_embeds: (Batch, Segments, 768)
        - padding_mask: (Batch, Segments) - True for padded positions
        - return_xai: (bool) - if True, returns dictionary with attention saliency and modality attribution
        """
        B, S, _ = text_embeds.size()
        
        # 1. Project both modalities through MLP ResBlocks
        t_proj = self.text_proj(text_embeds)   # (B, S, d_model)
        a_proj = self.audio_proj(audio_embeds) # (B, S, d_model)
        
        # 2. Cross-Modal Fusion
        if return_xai:
            fused, fusion_info = self.cross_modal_fusion(t_proj, a_proj, return_details=True)
        else:
            fused = self.cross_modal_fusion(t_proj, a_proj)
        
        # 3. Prepend [CLS] token: (B, 1 + S, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        seq_with_cls = torch.cat([cls_tokens, fused], dim=1)
        
        # 4. Add positional encoding
        h = self.pos_encoder(seq_with_cls)
        
        # 5. Adjust padding mask for [CLS] token (CLS is never padded -> False)
        if padding_mask is not None:
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=padding_mask.device)
            mask_with_cls = torch.cat([cls_mask, padding_mask], dim=1)
        else:
            mask_with_cls = None

        last_attn = None
        if return_xai:
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
        else:
            if mask_with_cls is not None:
                seq_out = self.sequence_transformer(h, src_key_padding_mask=mask_with_cls)
            else:
                seq_out = self.sequence_transformer(h)
            
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
            t_norm = torch.norm(t_out, p=2, dim=-1) # (B, S)
            a_norm = torch.norm(a_out, p=2, dim=-1) # (B, S)
            mean_t = t_norm.mean(dim=-1, keepdim=True)
            mean_a = a_norm.mean(dim=-1, keepdim=True)
            text_ratio = (mean_t / torch.clamp(mean_t + mean_a, min=1e-9)).squeeze(-1)
            audio_ratio = 1.0 - text_ratio

            return {
                "logits": logits,
                "probabilities": probs,
                "predicted_class": logits.argmax(dim=-1).item(),
                "turn_saliency": turn_saliency.squeeze(0),
                "text_ratio": float(text_ratio[0].item()),
                "audio_ratio": float(audio_ratio[0].item()),
                "text_norms": t_norm.squeeze(0),
                "audio_norms": a_norm.squeeze(0),
                "turn_cross_mismatch": (torch.abs(t_norm - a_norm) / torch.clamp(t_norm + a_norm, min=1e-9)).squeeze(0)
            }

        return logits, None, None


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
