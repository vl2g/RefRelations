"""
models/relationship_embedding.py

Query-conditioned Relationship Embedding (Section 3.2).

Implements:
  • Subject encoder  f_s(B_i^j)  (Eq. 4)
  • Object  encoder  f_o(B_i^j)  (Eq. 5)
  • GSA / LLA enrichment         (Eqs. 6-14)
  • Translational relation embed f_r(u_i^{jk})  (Eq. 15)
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregation import GlobalSemanticAggregation, LocalLocalizationAggregation


class SubjectObjectEncoder(nn.Module):
    """
    Encodes a single bounding-box candidate into a subject or object
    representation by fusing ROI appearance, spatial, and word features.

    Eq. 4 (subject) / Eq. 5 (object):
        f_s = W_s2 * ReLU(W_s1 * [f_app; f_spa; G(s)])
    """

    def __init__(
        self,
        roi_dim: int   = 2048,
        spa_dim: int   = 4,
        word_dim: int  = 300,
        hidden_dim: int = 512,
        out_dim: int   = 256,
    ) -> None:
        super().__init__()
        in_dim = roi_dim + spa_dim + word_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        roi_feat:  torch.Tensor,   # (..., 2048)
        spa_feat:  torch.Tensor,   # (..., 4)
        word_emb:  torch.Tensor,   # (..., 300)
    ) -> torch.Tensor:             # (..., out_dim)
        x = torch.cat([roi_feat, spa_feat, word_emb], dim=-1)
        return self.fc2(F.relu(self.fc1(x)))


class RelationshipEmbedding(nn.Module):
    """
    Full pipeline: encode subject & object, apply GSA / LLA, then compute
    translational relationship embedding (Eq. 15).

    Args:
        roi_dim      : FasterRCNN ROI feature dimension (2048)
        spa_dim      : Spatial feature dimension (4)
        word_dim     : GloVe embedding dimension (300)
        i3d_dim      : I3D feature dimension (1024)
        so_hidden    : Hidden dim for subject/object MLP
        so_out_dim   : Output dim for subject/object repr (256)
        rel_dim      : Translational embedding dim (256)
        use_gsa      : Whether to apply Global Semantic Aggregation
        use_lla      : Whether to apply Local Localization Aggregation
    """

    def __init__(
        self,
        roi_dim: int   = 2048,
        spa_dim: int   = 4,
        word_dim: int  = 300,
        i3d_dim: int   = 1024,
        so_hidden: int = 512,
        so_out_dim: int = 256,
        rel_dim: int   = 256,
        use_gsa: bool  = True,
        use_lla: bool  = True,
    ) -> None:
        super().__init__()
        self.use_gsa = use_gsa
        self.use_lla = use_lla
        self.so_out_dim = so_out_dim

        # Eq. 4 / 5: subject and object encoders (separate weights)
        self.subj_encoder = SubjectObjectEncoder(roi_dim, spa_dim, word_dim, so_hidden, so_out_dim)
        self.obj_encoder  = SubjectObjectEncoder(roi_dim, spa_dim, word_dim, so_hidden, so_out_dim)

        # GSA modules (one per subject, one per object) – Eqs. 6-10
        if use_gsa:
            self.gsa_subj = GlobalSemanticAggregation(i3d_dim, word_dim, so_out_dim, so_hidden)
            self.gsa_obj  = GlobalSemanticAggregation(i3d_dim, word_dim, so_out_dim, so_hidden)

        # LLA modules – Eqs. 11-12
        if use_lla:
            roi_spa_dim = roi_dim + spa_dim
            self.lla_subj = LocalLocalizationAggregation(roi_spa_dim, so_out_dim, window=5, hidden_dim=so_hidden)
            self.lla_obj  = LocalLocalizationAggregation(roi_spa_dim, so_out_dim, window=5, hidden_dim=so_hidden)

        # Eqs. 13-14: fuse GSA and LLA representations
        self.fuse_subj = nn.Sequential(
            nn.Linear(so_out_dim * 2, so_hidden), nn.ReLU(),
            nn.Linear(so_hidden, so_out_dim),
        )
        self.fuse_obj = nn.Sequential(
            nn.Linear(so_out_dim * 2, so_hidden), nn.ReLU(),
            nn.Linear(so_hidden, so_out_dim),
        )

        # Eq. 15: translational relationship embedding
        # input: [W_rs * f_j − W_ro * f_k; G(p)]
        self.W_rs = nn.Linear(so_out_dim, so_out_dim, bias=False)
        self.W_ro = nn.Linear(so_out_dim, so_out_dim, bias=False)
        self.rel_fc1 = nn.Linear(so_out_dim + word_dim, so_hidden)
        self.rel_fc2 = nn.Linear(so_hidden, rel_dim)

    # ------------------------------------------------------------------ #

    def encode_so(
        self,
        roi_feats:  torch.Tensor,   # (T, M, 2048)
        spatial:    torch.Tensor,   # (T, M, 4)
        i3d_feats:  torch.Tensor,   # (T, 1024)
        subj_emb:   torch.Tensor,   # (300,)
        obj_emb:    torch.Tensor,   # (300,)
    ):
        """
        Returns enriched subject and object representations for every
        (frame, box) pair: (T, M, so_out_dim) each.
        """
        T, M, _ = roi_feats.shape

        # Expand word embeddings to match (T, M, 300)
        subj_emb_exp = subj_emb.unsqueeze(0).unsqueeze(0).expand(T, M, -1)
        obj_emb_exp  = obj_emb.unsqueeze(0).unsqueeze(0).expand(T, M, -1)

        # Eqs. 4 & 5: base representations
        fs_base = self.subj_encoder(roi_feats, spatial, subj_emb_exp)  # (T, M, D)
        fo_base = self.obj_encoder(roi_feats,  spatial, obj_emb_exp)   # (T, M, D)

        # ── GSA (Eqs. 6-10): operates at video level ─────────────────── #
        if self.use_gsa:
            # Compute one global context per subject & object representation
            # We process box-by-box for clarity (can be vectorised)
            fs_gsa_list = []
            fo_gsa_list = []
            for m in range(M):
                fs_m_gsa = torch.stack(
                    [self.gsa_subj(i3d_feats, subj_emb, fs_base[t, m])
                     for t in range(T)]
                )  # (T, D)
                fo_m_gsa = torch.stack(
                    [self.gsa_obj(i3d_feats, obj_emb, fo_base[t, m])
                     for t in range(T)]
                )  # (T, D)
                fs_gsa_list.append(fs_m_gsa)
                fo_gsa_list.append(fo_m_gsa)
            fs_gsa = torch.stack(fs_gsa_list, dim=1)  # (T, M, D)
            fo_gsa = torch.stack(fo_gsa_list, dim=1)  # (T, M, D)
        else:
            fs_gsa = fs_base
            fo_gsa = fo_base

        # ── LLA (Eqs. 11-12): operates at frame level ─────────────────── #
        if self.use_lla:
            roi_spa = torch.cat([roi_feats, spatial], dim=-1)  # (T, M, 2052)
            fs_lla_list = []
            fo_lla_list = []
            for m in range(M):
                fs_m_lla = torch.stack(
                    [self.lla_subj(roi_spa[:, m, :], fs_base[t, m], t)
                     for t in range(T)]
                )
                fo_m_lla = torch.stack(
                    [self.lla_obj(roi_spa[:, m, :], fo_base[t, m], t)
                     for t in range(T)]
                )
                fs_lla_list.append(fs_m_lla)
                fo_lla_list.append(fo_m_lla)
            fs_lla = torch.stack(fs_lla_list, dim=1)  # (T, M, D)
            fo_lla = torch.stack(fo_lla_list, dim=1)  # (T, M, D)
        else:
            fs_lla = fs_base
            fo_lla = fo_base

        # ── Eqs. 13-14: fuse GSA and LLA ─────────────────────────────── #
        fs = self.fuse_subj(torch.cat([fs_gsa, fs_lla], dim=-1))  # (T, M, D)
        fo = self.fuse_obj( torch.cat([fo_gsa, fo_lla], dim=-1))  # (T, M, D)

        return fs, fo  # (T, M, so_out_dim) each

    def compute_rel_embedding(
        self,
        fs:       torch.Tensor,   # (T, M, so_out_dim)  subject features
        fo:       torch.Tensor,   # (T, M, so_out_dim)  object  features
        pred_emb: torch.Tensor,   # (300,)               predicate GloVe
    ) -> torch.Tensor:            # (T, M, M, rel_dim)
        """
        Eq. 15: f_r(u_i^{jk}) = W_r2 ReLU W_r1 ReLU ([W_rs f_j − W_ro f_k; G(p)])
        Computes the relation embedding for every ordered pair (j, k) with j≠k.
        """
        T, M, D = fs.shape

        # Project subject and object features
        fs_proj = self.W_rs(fs)  # (T, M, D)
        fo_proj = self.W_ro(fo)  # (T, M, D)

        # Expand for pairwise computation: (T, M, 1, D) - (T, 1, M, D)
        diff = fs_proj.unsqueeze(2) - fo_proj.unsqueeze(1)  # (T, M, M, D)

        # Concatenate predicate embedding
        p_exp = pred_emb.view(1, 1, 1, -1).expand(T, M, M, -1)
        x = torch.cat([diff, p_exp], dim=-1)  # (T, M, M, D + 300)

        # Two-layer MLP
        out = self.rel_fc2(F.relu(self.rel_fc1(x)))  # (T, M, M, rel_dim)
        return out

    def forward(
        self,
        roi_feats:  torch.Tensor,   # (T, M, 2048)
        spatial:    torch.Tensor,   # (T, M, 4)
        i3d_feats:  torch.Tensor,   # (T, 1024)
        subj_emb:   torch.Tensor,   # (300,)
        obj_emb:    torch.Tensor,   # (300,)
        pred_emb:   torch.Tensor,   # (300,)
    ) -> torch.Tensor:              # (T, M, M, rel_dim)
        fs, fo = self.encode_so(roi_feats, spatial, i3d_feats, subj_emb, obj_emb)
        return self.compute_rel_embedding(fs, fo, pred_emb)
