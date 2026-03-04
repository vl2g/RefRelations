"""
datasets/vidvrd_dataset.py
ImageNet-VidVRD dataset loader for few-shot referring relationships.

Dataset:  https://xdshang.github.io/docs/imagenet-vidvrd.html
Structure:
    vidvrd/
        videos/           ← raw .mp4 files
        annotations/      ← per-video JSON files
        features/         ← pre-extracted (see scripts/extract_features.py)
"""
from __future__ import annotations
import json
import os
import random
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch

from .base_dataset import VideoRelationDataset

_TRAIN_PREDICATES = [
    "stop", "front", "taller", "right", "larger", "stand front", "sit behind",
    "above", "left", "walk behind", "run beneath", "move right", "play", "feed",
    "faster", "move front", "fly above", "next to", "away", "creep toward",
    "follow", "in front of", "behind", "beside", "watch", "toward", "chase",
    "hold", "kick", "bite", "hit", "touch", "carry", "eat", "catch", "pull",
    "push", "throw", "jump over", "jump", "fly", "walk", "run", "stand",
    "sit", "lie", "swim", "ride", "drive", "move", "stop", "fall off",
    "lean on", "hang on", "trot", "gallop", "sniff", "lick", "attack",
    "fight", "play with", "interact with", "look at", "talk to", "wave to",
    "point at", "lift", "pick up", "put down", "open", "close", "turn",
    "spin", "roll", "slide", "bounce", "swing", "shake", "stretch",
    "lean", "crouch", "kneel", "climb", "descend", "enter", "exit",
    "cross", "pass", "approach",
]

_TEST_PREDICATES = [
    "fly with", "walk with", "stand with", "sit with", "run with",
    "play with animal", "feed animal", "ride animal", "herd", "train",
    "direct", "lead", "escort", "guard", "patrol",
    "gather", "disperse", "surround", "block", "merge",
    "overtake", "collide",
]


class VidVRDDataset(VideoRelationDataset):
    """ImageNet-VidVRD few-shot episode dataset."""

    def _build_index(self) -> None:
        ann_dir = os.path.join(self.data_root, "annotations")
        if not os.path.isdir(ann_dir):
            print(f"[WARNING] Annotation directory not found: {ann_dir}")
            return

        for fname in sorted(os.listdir(ann_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(ann_dir, fname)
            with open(fpath) as f:
                ann = json.load(f)

            video_id = ann.get("video_id", fname.replace(".json", ""))
            for rel in ann.get("relation_instances", []):
                pred = rel["predicate"]
                subj = rel["subject_tid"]
                obj  = rel["object_tid"]

                # Resolve trajectory indices → category names
                tid2cat = {t["tid"]: t["category"]
                           for t in ann.get("subject/objects", [])}
                subj_name = tid2cat.get(subj, "unknown")
                obj_name  = tid2cat.get(obj,  "unknown")

                entry = {
                    "video_id":    video_id,
                    "subject":     subj_name,
                    "subject_tid": subj,
                    "object":      obj_name,
                    "object_tid":  obj,
                    "predicate":   pred,
                    "begin_fid":   rel.get("begin_fid", 0),
                    "end_fid":     rel.get("end_fid", -1),
                }
                self.predicate2videos.setdefault(pred, []).append(entry)

    def _split_predicates(self) -> Tuple[List[str], List[str]]:
        # Use paper split if available; otherwise fall back to stored lists
        all_preds = list(self.predicate2videos.keys())
        train = [p for p in all_preds if p in _TRAIN_PREDICATES]
        test  = [p for p in all_preds if p in _TEST_PREDICATES]
        # If annotations not loaded yet, return hardcoded lists
        if not train:
            train = _TRAIN_PREDICATES
        if not test:
            test = _TEST_PREDICATES
        return train, test

    def _load_video_features(self, video_id: str) -> Dict[str, torch.Tensor]:
        """Load pre-extracted features from HDF5 file."""
        h5_path = os.path.join(self.feature_root, f"{video_id}.h5")

        if not os.path.isfile(h5_path):
            # Return dummy tensors so the pipeline can run without data
            T, M = self.max_frames, self.num_boxes
            return self._dummy_features(T, M)

        with h5py.File(h5_path, "r") as hf:
            roi_feats  = torch.tensor(hf["roi_feats"][:],  dtype=torch.float32)
            spatial    = torch.tensor(hf["spatial"][:],    dtype=torch.float32)
            i3d_feats  = torch.tensor(hf["i3d_feats"][:], dtype=torch.float32)
            gt_subject = torch.tensor(hf["gt_subject"][:], dtype=torch.float32)
            gt_object  = torch.tensor(hf["gt_object"][:],  dtype=torch.float32)

        return {
            "roi_feats":  roi_feats,   # (T, M, 2048)
            "spatial":    spatial,     # (T, M, 4)
            "i3d_feats":  i3d_feats,   # (T, 1024)
            "gt_subject": gt_subject,  # (T, 4)
            "gt_object":  gt_object,   # (T, 4)
        }

    @staticmethod
    def _dummy_features(T: int, M: int) -> Dict[str, torch.Tensor]:
        return {
            "roi_feats":  torch.zeros(T, M, 2048),
            "spatial":    torch.zeros(T, M, 4),
            "i3d_feats":  torch.zeros(T, 1024),
            "gt_subject": torch.zeros(T, 4),
            "gt_object":  torch.zeros(T, 4),
        }
