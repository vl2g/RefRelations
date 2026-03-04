"""
datasets/vidor_dataset.py
VidOR dataset loader for few-shot referring relationships.

Dataset:  https://xdshang.github.io/docs/vidor.html
10,000 videos | 50 predicates | 80 object categories
"""
from __future__ import annotations
import json
import os
from typing import Dict, List, Tuple

import h5py
import torch

from .base_dataset import VideoRelationDataset

_TRAIN_PREDICATES = [
    "watch", "bite", "carry", "hold", "touch", "ride", "drive", "pull",
    "push", "throw", "kick", "hit", "feed", "play with", "chase",
    "follow", "lean on", "hang on", "hold hand of", "hug", "kiss",
    "pat", "point to", "wave", "lift", "pick up", "put down",
    "open", "close", "wave to", "talk to", "look at",
    "walk beside", "run beside", "stand beside",
]

_TEST_PREDICATES = [
    "sit next to", "stand behind", "walk in front of", "run in front of",
    "fly above", "above", "next to", "in front of", "behind",
    "toward", "away", "past", "stop", "fall off", "jump over",
]


class VidORDataset(VideoRelationDataset):
    """VidOR few-shot episode dataset."""

    def _build_index(self) -> None:
        ann_dir = os.path.join(self.data_root, "annotations")
        if not os.path.isdir(ann_dir):
            print(f"[WARNING] VidOR annotation directory not found: {ann_dir}")
            return

        # VidOR annotations are nested: annotations/<group>/<video_id>.json
        for grp in sorted(os.listdir(ann_dir)):
            grp_path = os.path.join(ann_dir, grp)
            if not os.path.isdir(grp_path):
                continue
            for fname in sorted(os.listdir(grp_path)):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(grp_path, fname)
                with open(fpath) as f:
                    ann = json.load(f)

                video_id = ann.get("video_id", fname.replace(".json", ""))
                tid2cat = {
                    str(obj["tid"]): obj["category"]
                    for obj in ann.get("subject/objects", [])
                }

                for rel in ann.get("relation_instances", []):
                    pred = rel["predicate"]
                    subj_tid = str(rel["subject_tid"])
                    obj_tid  = str(rel["object_tid"])
                    entry = {
                        "video_id":    video_id,
                        "subject":     tid2cat.get(subj_tid, "unknown"),
                        "subject_tid": subj_tid,
                        "object":      tid2cat.get(obj_tid,  "unknown"),
                        "object_tid":  obj_tid,
                        "predicate":   pred,
                        "begin_fid":   rel.get("begin_fid", 0),
                        "end_fid":     rel.get("end_fid", -1),
                    }
                    self.predicate2videos.setdefault(pred, []).append(entry)

    def _split_predicates(self) -> Tuple[List[str], List[str]]:
        all_preds = list(self.predicate2videos.keys())
        train = [p for p in all_preds if p in _TRAIN_PREDICATES] or _TRAIN_PREDICATES
        test  = [p for p in all_preds if p in _TEST_PREDICATES]  or _TEST_PREDICATES
        return train, test

    def _load_video_features(self, video_id: str) -> Dict[str, torch.Tensor]:
        h5_path = os.path.join(self.feature_root, f"{video_id}.h5")
        if not os.path.isfile(h5_path):
            T, M = self.max_frames, self.num_boxes
            return {
                "roi_feats":  torch.zeros(T, M, 2048),
                "spatial":    torch.zeros(T, M, 4),
                "i3d_feats":  torch.zeros(T, 1024),
                "gt_subject": torch.zeros(T, 4),
                "gt_object":  torch.zeros(T, 4),
            }
        with h5py.File(h5_path, "r") as hf:
            return {
                "roi_feats":  torch.tensor(hf["roi_feats"][:],  dtype=torch.float32),
                "spatial":    torch.tensor(hf["spatial"][:],    dtype=torch.float32),
                "i3d_feats":  torch.tensor(hf["i3d_feats"][:], dtype=torch.float32),
                "gt_subject": torch.tensor(hf["gt_subject"][:], dtype=torch.float32),
                "gt_object":  torch.tensor(hf["gt_object"][:],  dtype=torch.float32),
            }
