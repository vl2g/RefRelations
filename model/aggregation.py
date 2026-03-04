"""
models/aggregation.py

Global Semantic Aggregation (GSA) and Local Localization Aggregation (LLA)
as described in Section 3.2 of the paper.

GSA  – fuses I3D frame features weighted by an attention vector conditioned
       on the GloVe word embedding of the subject / object.

LLA  – adds local temporal context from ±2 adjacent frames (window = 5).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Global Semantic Aggregation (Eqs. 6-10)
# ─────────────────────────────────────────────────────────────────────────────

class GlobalSemanticAggregation(nn.Module):
    """
    Computes a global attention vector over all T frames using I3D features
    conditioned on the word embedding of the subject or object.

    Args:
        i3d_dim   : dimension of I3D per-frame feature   (default 1024)
        word_dim  : GloVe embedding dimension             (default 300)
        so_dim    : subject/object representation dim     (default 256)
        hidden_dim: hidden size of attention MLP
    """

    def __init__(
        self,
        i3d_dim: int = 1024,
        word_dim: int = 300,
        so_dim: int = 256,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        # Eq. 7: s_j^g = W_gs1 * ReLU(W_gs2 * [f_I3D; G(s)] + b_gs)
        self.att_fc1 = nn.Linear(i3d_dim + word_dim, hidden_dim)
        self.att_fc2 = nn.Linear(hidden_dim, 1)
        # Eq. 9: f_s^g = sum_j (alpha_j ⊙ f_I3D_j) · f_s
        # We fuse via a linear projection: [attended_i3d; f_so] → f_so_dim
        self.fusion_fc = nn.Linear(i3d_dim + so_dim, so_dim)

    def forward(
        self,
        i3d_feats: torch.Tensor,   # (T, i3d_dim)
        word_emb:  torch.Tensor,   # (word_dim,) or (1, word_dim)
        so_feat:   torch.Tensor,   # (so_dim,)   or (1, so_dim)
    ) -> torch.Tensor:             # (so_dim,)
        T = i3d_feats.size(0)
        # Expand word embedding to (T, word_dim)
        w = word_emb.unsqueeze(0).expand(T, -1)           # (T, word_dim)
        x = torch.cat([i3d_feats, w], dim=-1)             # (T, i3d_dim+word_dim)
        s = self.att_fc2(F.relu(self.att_fc1(x)))         # (T, 1)  Eq. 7
        alpha = torch.softmax(s, dim=0)                   # (T, 1)  Eq. 8
        # Attended I3D: (i3d_dim,)
        attended = (alpha * i3d_feats).sum(dim=0)         # Eq. 9  (i3d_dim,)
        # Fuse with original subject/object representation
        fused = self.fusion_fc(
            torch.cat([attended, so_feat], dim=-1)
        )                                                  # (so_dim,)
        return fused


# ─────────────────────────────────────────────────────────────────────────────
# Local Localization Aggregation (Eqs. 11-12)
# ─────────────────────────────────────────────────────────────────────────────

class LocalLocalizationAggregation(nn.Module):
    """
    Adds local temporal context from a window of ±2 frames around frame i
    to improve robustness under occlusion / partial visibility.

    Args:
        roi_dim   : ROI appearance + spatial feature dim per box
        so_dim    : subject/object representation dim
        window    : must be 5 (i−2 … i+2); kept as param for clarity
    """

    def __init__(
        self,
        roi_dim: int = 2052,    # 2048 app + 4 spatial
        so_dim: int = 256,
        window: int = 5,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.window = window
        half = window // 2  # = 2

        # Eq. 12: difference-based local feature
        # input: (f_{t-2} - f_{t-1}) * (f_{t+1} - f_{t+2})  → roi_dim
        self.local_fc1 = nn.Linear(roi_dim, hidden_dim)
        self.local_fc2 = nn.Linear(hidden_dim, so_dim)

        # Attention: LAtt(f_l, f_s)  → scalar weight
        self.att_fc1 = nn.Linear(so_dim + so_dim, hidden_dim)
        self.att_fc2 = nn.Linear(hidden_dim, 1)

        # Final fusion
        self.fusion_fc = nn.Linear(so_dim + so_dim, so_dim)

    def forward(
        self,
        stacked_roi: torch.Tensor,  # (T, roi_dim) – stacked ROI+spatial for one box
        so_feat:     torch.Tensor,  # (so_dim,) – current frame's subject/object repr
        frame_idx:   int,
    ) -> torch.Tensor:              # (so_dim,)
        T = stacked_roi.size(0)
        half = self.window // 2

        # Pad with replicated boundary frames
        pad_before = stacked_roi[0:1].expand(half, -1)
        pad_after  = stacked_roi[-1:].expand(half, -1)
        padded = torch.cat([pad_before, stacked_roi, pad_after], dim=0)
        # Now index i corresponds to padded[i + half]
        i = frame_idx + half

        f_tm2 = padded[i - 2]
        f_tm1 = padded[i - 1]
        f_tp1 = padded[i + 1]
        f_tp2 = padded[i + 2]

        # Eq. 12: element-wise product of first and second differences
        local_feat = (f_tm2 - f_tm1) * (f_tp1 - f_tp2)    # (roi_dim,)
        local_repr = self.local_fc2(F.relu(self.local_fc1(local_feat)))  # (so_dim,)

        # Attention weight (Eq. 11)
        att_in = torch.cat([local_repr, so_feat], dim=-1)
        alpha  = torch.sigmoid(self.att_fc2(F.relu(self.att_fc1(att_in))))  # (1,)

        # Weighted fusion
        fused = self.fusion_fc(
            torch.cat([alpha * local_repr, so_feat], dim=-1)
        )
        return fused
