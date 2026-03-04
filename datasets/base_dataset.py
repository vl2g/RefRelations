"""
datasets/base_dataset.py
Abstract base class for VidVRD / VidOR datasets.
"""
from __future__ import annotations
import os
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class VideoRelationDataset(ABC, Dataset):
    """
    Abstract base for few-shot video visual relationship datasets.

    Each item is an *episode*:
        - support set  : K videos sharing query predicate p
        - test video   : one video containing <s, p, o>
        - query        : (subject_name, predicate_name, object_name)
    """

    def __init__(
        self,
        data_root: str,
        feature_root: str,
        glove_path: str,
        split: str = "train",          # "train" | "test"
        support_size: int = 4,
        num_boxes: int = 30,
        max_frames: int = 30,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.feature_root = feature_root
        self.split = split
        self.support_size = support_size
        self.num_boxes = num_boxes        # M
        self.max_frames = max_frames      # T cap for GPU memory

        # Build predicate / video index (implemented by subclass)
        self.predicate2videos: Dict[str, List[dict]] = {}
        self.episodes: List[dict] = []    # populated by _build_episodes()

        # Load GloVe
        self.glove = self._load_glove(glove_path)
        self._build_index()
        self._build_episodes()

    # ------------------------------------------------------------------ #
    #  Abstract methods                                                    #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _build_index(self) -> None:
        """Populate self.predicate2videos from annotation files."""

    @abstractmethod
    def _load_video_features(self, video_id: str) -> Dict[str, torch.Tensor]:
        """
        Return dict with keys:
            roi_feats  : (T, M, 2048)  – FasterRCNN appearance
            spatial    : (T, M, 4)     – normalised bbox coords
            i3d_feats  : (T, 1024)     – I3D per-frame features
            gt_subject : (T, 4)        – ground-truth subject bbox per frame
            gt_object  : (T, 4)        – ground-truth object  bbox per frame
        """

    @abstractmethod
    def _split_predicates(self) -> Tuple[List[str], List[str]]:
        """Return (train_predicates, test_predicates)."""

    # ------------------------------------------------------------------ #
    #  Episode building                                                    #
    # ------------------------------------------------------------------ #

    def _build_episodes(self) -> None:
        train_preds, test_preds = self._split_predicates()
        preds = train_preds if self.split == "train" else test_preds

        for pred in preds:
            videos = self.predicate2videos.get(pred, [])
            if len(videos) < self.support_size + 1:
                continue
            # Each video can be the test video once; support drawn from rest
            for i, test_vid in enumerate(videos):
                pool = [v for j, v in enumerate(videos) if j != i]
                support = random.sample(pool, min(self.support_size, len(pool)))
                self.episodes.append(
                    {
                        "predicate": pred,
                        "test_video": test_vid,
                        "support_videos": support,
                    }
                )

    # ------------------------------------------------------------------ #
    #  Dataset interface                                                   #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        ep = self.episodes[idx]

        # Test video features
        test_feats = self._load_video_features(ep["test_video"]["video_id"])
        test_feats = self._cap_frames(test_feats)

        # Support set features
        support_feats = []
        for sv in ep["support_videos"]:
            sf = self._load_video_features(sv["video_id"])
            sf = self._cap_frames(sf)
            sf["subject_name"] = sv["subject"]
            sf["object_name"] = sv["object"]
            support_feats.append(sf)

        # Query word embeddings
        query_subject_emb = self._word_embed(ep["test_video"]["subject"])
        query_object_emb  = self._word_embed(ep["test_video"]["object"])
        query_pred_emb    = self._word_embed(ep["predicate"])

        return {
            "test_feats": test_feats,
            "support_feats": support_feats,
            "query": {
                "subject":    ep["test_video"]["subject"],
                "predicate":  ep["predicate"],
                "object":     ep["test_video"]["object"],
                "subject_emb": query_subject_emb,
                "object_emb":  query_object_emb,
                "pred_emb":    query_pred_emb,
            },
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _cap_frames(self, feats: dict) -> dict:
        """Uniformly sample up to max_frames from a video's features."""
        T = feats["roi_feats"].shape[0]
        if T <= self.max_frames:
            return feats
        idx = np.linspace(0, T - 1, self.max_frames, dtype=int)
        feats["roi_feats"]  = feats["roi_feats"][idx]
        feats["spatial"]    = feats["spatial"][idx]
        feats["i3d_feats"]  = feats["i3d_feats"][idx]
        feats["gt_subject"] = feats["gt_subject"][idx]
        feats["gt_object"]  = feats["gt_object"][idx]
        return feats

    def _word_embed(self, word: str) -> torch.Tensor:
        """Return 300-d GloVe embedding; fall back to zeros if OOV."""
        token = word.lower().replace(" ", "_")
        vec = self.glove.get(token, np.zeros(300, dtype=np.float32))
        return torch.tensor(vec, dtype=torch.float32)

    @staticmethod
    def _load_glove(path: str) -> Dict[str, np.ndarray]:
        """Load GloVe text file into a dict."""
        glove: Dict[str, np.ndarray] = {}
        if not os.path.isfile(path):
            print(f"[WARNING] GloVe file not found at {path}. Using zero vectors.")
            return glove
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                glove[parts[0]] = np.array(parts[1:], dtype=np.float32)
        return glove
