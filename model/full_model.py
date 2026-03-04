"""
models/full_model.py

Top-level model that ties together:
  RelationshipEmbedding + RelationNetwork + Random Field solver.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .relationship_embedding import RelationshipEmbedding
from .relation_network import RelationNetwork, EpisodicLoss
from .random_field import generate_trajectories


class FewShotReferringRelationship(nn.Module):
    """
    Few-Shot Referring Relationship model (Kumar & Mishra, CVPR 2023).

    Forward pass returns relationship embeddings for the test video and
    (during eval) predicted bounding-box index trajectories.
    """

    def __init__(
        self,
        roi_dim: int    = 2048,
        spa_dim: int    = 4,
        word_dim: int   = 300,
        i3d_dim: int    = 1024,
        so_hidden: int  = 512,
        so_out_dim: int = 256,
        rel_dim: int    = 256,
        rn_hidden: int  = 512,
        use_gsa: bool   = True,
        use_lla: bool   = True,
        bp_iterations: int   = 10,
        bp_threshold: float  = -0.5,
        solver: str          = "belief_propagation",
    ) -> None:
        super().__init__()

        self.rel_emb_net = RelationshipEmbedding(
            roi_dim=roi_dim, spa_dim=spa_dim, word_dim=word_dim,
            i3d_dim=i3d_dim, so_hidden=so_hidden, so_out_dim=so_out_dim,
            rel_dim=rel_dim, use_gsa=use_gsa, use_lla=use_lla,
        )
        self.relation_net = RelationNetwork(rel_dim=rel_dim, hidden_dim=rn_hidden)
        self.loss_fn      = EpisodicLoss()

        self.bp_iterations = bp_iterations
        self.bp_threshold  = bp_threshold
        self.solver        = solver

    # ------------------------------------------------------------------ #
    #  Embed a single video                                               #
    # ------------------------------------------------------------------ #

    def embed_video(self, feats: dict, query: dict) -> torch.Tensor:
        """Returns (T, M, M, rel_dim) relationship embedding for a video."""
        return self.rel_emb_net(
            roi_feats=feats["roi_feats"],
            spatial=feats["spatial"],
            i3d_feats=feats["i3d_feats"],
            subj_emb=query["subject_emb"],
            obj_emb=query["object_emb"],
            pred_emb=query["pred_emb"],
        )

    # ------------------------------------------------------------------ #
    #  Training forward                                                   #
    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        support_feats: List[dict],
        query: dict,
    ) -> torch.Tensor:
        """
        Extract ground-truth positive and negative relationship embeddings
        from the support set and compute the episodic loss (Eq. 18).
        """
        pos_embs, neg_embs = [], []

        for sf in support_feats:
            emb = self.embed_video(sf, query)   # (T, M, M, D)
            T, M, _, D = emb.shape

            # Positive: GT subject & object indices (if annotated)
            # Here we use the centre-frame GT box indices as proxy
            mid = T // 2
            # For support videos, ground truth is known
            # We use a simple heuristic: IoU-matched box index
            gt_j, gt_k = self._gt_box_indices(sf, mid, M)
            if gt_j >= 0 and gt_k >= 0:
                pos_embs.append(emb[mid, gt_j, gt_k])

            # Negatives: random pairs excluding GT
            for _ in range(min(3, M * M - 1)):
                j = torch.randint(0, M, ()).item()
                k = torch.randint(0, M, ()).item()
                while j == k or (j == gt_j and k == gt_k):
                    j = torch.randint(0, M, ()).item()
                    k = torch.randint(0, M, ()).item()
                neg_embs.append(emb[mid, j, k])

        if not pos_embs or not neg_embs:
            return torch.tensor(0.0, requires_grad=True)

        pos_stack = torch.stack(pos_embs)   # (N_pos, D)
        neg_stack = torch.stack(neg_embs)   # (N_neg, D)
        return self.loss_fn(self.relation_net, pos_stack, neg_stack)

    # ------------------------------------------------------------------ #
    #  Inference forward                                                  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict(
        self,
        test_feats:    dict,
        support_feats: List[dict],
        query:         dict,
        solver:        Optional[str] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Returns predicted subject and object bounding-box indices per frame.
        """
        test_emb = self.embed_video(test_feats, query)
        sup_embs  = [self.embed_video(sf, query) for sf in support_feats]

        subj_traj, obj_traj = generate_trajectories(
            rel_emb_test=test_emb,
            rel_emb_support=sup_embs,
            rn=self.relation_net,
            solver=solver or self.solver,
            bp_iterations=self.bp_iterations,
            threshold=self.bp_threshold,
        )
        return subj_traj, obj_traj

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _gt_box_indices(feats: dict, frame_idx: int, M: int) -> Tuple[int, int]:
        """
        Return the FasterRCNN box index that best matches the GT bbox at a frame.
        Uses maximum IoU matching.
        """
        gt_sub = feats.get("gt_subject")
        gt_obj = feats.get("gt_object")
        spatial = feats.get("spatial")   # (T, M, 4)

        if gt_sub is None or gt_obj is None or spatial is None:
            return 0, 1  # fallback

        t = min(frame_idx, gt_sub.shape[0] - 1)
        boxes = spatial[t]  # (M, 4)

        def best_match(gt_box):
            ious = _batch_iou(boxes, gt_box.unsqueeze(0))  # (M,)
            idx  = ious.argmax().item()
            return idx if ious[idx] > 0 else -1

        j = best_match(gt_sub[t])
        k = best_match(gt_obj[t])
        return j, k


def _batch_iou(
    boxes1: torch.Tensor,   # (N, 4)  [x1, y1, x2, y2] normalised
    boxes2: torch.Tensor,   # (M, 4)
) -> torch.Tensor:           # (N, M)
    """Compute pairwise IoU between two sets of boxes."""
    x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
    y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
    x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
    y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter   = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter

    return inter / union.clamp(min=1e-6)
