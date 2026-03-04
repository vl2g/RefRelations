"""
scripts/extract_features.py

Pre-extract FasterRCNN (ROI + spatial) and I3D features from raw videos
and save them as HDF5 files.

Usage:
    python scripts/extract_features.py \\
        --dataset vidvrd \\
        --data_root data/vidvrd \\
        --output_dir data/features/vidvrd \\
        --device cuda

Dependencies:
    - torchvision (FasterRCNN)
    - A pre-trained I3D checkpoint (pytorch-i3d or timm)

Notes:
    Each output HDF5 file has the following datasets:
        roi_feats  : float32  (T, M, 2048)
        spatial    : float32  (T, M, 4)     normalised [x1,y1,x2,y2] in [0,1]
        i3d_feats  : float32  (T, 1024)
        gt_subject : float32  (T, 4)        present only if annotations exist
        gt_object  : float32  (T, 4)
"""
from __future__ import annotations
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch
import torchvision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.ops import roi_pool
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# FasterRCNN feature extractor
# ─────────────────────────────────────────────────────────────────────────────

class FasterRCNNExtractor:
    """
    Extracts per-frame object proposals (boxes + ROI features) using a
    FasterRCNN-ResNet101 backbone (the paper uses ResNet-101 pre-trained on
    MS-COCO; here we default to ResNet-50 for convenience—swap the backbone
    if you have the R-101 weights).
    """

    def __init__(self, device: str = "cuda", num_boxes: int = 30) -> None:
        self.device = torch.device(device)
        self.num_boxes = num_boxes

        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        self.model.to(self.device).eval()

        # Hook into the backbone to extract ROI features
        self._feat_map: Optional[torch.Tensor] = None
        self.model.backbone.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        # output is an OrderedDict; '3' is the final P4 feature map
        self._feat_map = output["3"].detach()

    @torch.no_grad()
    def extract(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            frame: (H, W, 3) BGR uint8 from OpenCV

        Returns:
            roi_feats : (num_boxes, 2048)
            spatial   : (num_boxes, 4)  normalised [x1,y1,x2,y2]
        """
        H, W = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_t   = torch.from_numpy(img_rgb).float().div(255.0)
        img_t   = img_t.permute(2, 0, 1).unsqueeze(0).to(self.device)

        predictions = self.model(img_t)[0]
        boxes  = predictions["boxes"]     # (N, 4) in pixel coords
        scores = predictions["scores"]    # (N,)

        # Keep top-M boxes
        keep = scores.argsort(descending=True)[: self.num_boxes]
        boxes = boxes[keep]               # (M', 4)
        M     = boxes.size(0)

        # Pad if fewer than num_boxes detections
        if M < self.num_boxes:
            pad  = torch.zeros(self.num_boxes - M, 4, device=self.device)
            boxes = torch.cat([boxes, pad], dim=0)
            M = self.num_boxes

        # ROI-pool from the backbone feature map  → (M, 2048, 1, 1)
        scaled_boxes = boxes.clone()
        feat_stride  = img_t.shape[-1] / self._feat_map.shape[-1]
        scaled_boxes = scaled_boxes / feat_stride
        rois = torch.cat([
            torch.zeros(M, 1, device=self.device),   # batch index = 0
            scaled_boxes
        ], dim=1)
        roi_feats_4d = roi_pool(self._feat_map, rois, output_size=(1, 1))  # (M,C,1,1)
        roi_feats    = roi_feats_4d.squeeze(-1).squeeze(-1).cpu().numpy()  # (M, C)

        # Normalise box coordinates
        spatial       = boxes.cpu().numpy()
        spatial[:, 0] /= W; spatial[:, 2] /= W
        spatial[:, 1] /= H; spatial[:, 3] /= H

        return roi_feats, spatial


# ─────────────────────────────────────────────────────────────────────────────
# I3D feature extractor (stub – replace with full I3D implementation)
# ─────────────────────────────────────────────────────────────────────────────

class I3DExtractor:
    """
    Extracts per-clip I3D features.
    Replace `_load_model` with your pre-trained I3D checkpoint.
    """

    def __init__(self, device: str = "cuda", clip_len: int = 16) -> None:
        self.device = torch.device(device)
        self.clip_len = clip_len
        self.model = self._load_model()

    def _load_model(self):
        """
        Load a pre-trained I3D model.
        Example using the pytorch-i3d library:

            from i3d_pytorch import InceptionI3d
            model = InceptionI3d(400, in_channels=3)
            model.load_state_dict(torch.load('rgb_imagenet.pt'))
            model.to(self.device).eval()
            return model

        Here we return a dummy nn.Identity for skeleton compatibility.
        """
        import torch.nn as nn
        dummy = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        return dummy

    @torch.no_grad()
    def extract(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Args:
            frames: list of (H, W, 3) BGR uint8 frames for one clip

        Returns:
            feat: (1024,) I3D feature vector
        """
        if not frames:
            return np.zeros(1024, dtype=np.float32)

        clip = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])
        clip = clip.astype(np.float32) / 255.0
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0)  # (1,3,T,H,W)
        clip = clip.to(self.device)
        out  = self.model(clip)
        return out.squeeze().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction loop
# ─────────────────────────────────────────────────────────────────────────────

def extract_video(
    video_path: str,
    output_path: str,
    frcnn: FasterRCNNExtractor,
    i3d:   I3DExtractor,
    gt_annotations: Optional[dict] = None,
    num_boxes: int = 30,
    frame_step: int = 4,   # sample every N frames to reduce redundancy
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return

    frames_roi  = []
    frames_spa  = []
    frames_i3d_raw = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            roi, spa = frcnn.extract(frame)
            frames_roi.append(roi)
            frames_spa.append(spa)
            frames_i3d_raw.append(frame)
        frame_idx += 1
    cap.release()

    if not frames_roi:
        return

    roi_feats = np.stack(frames_roi)   # (T, M, 2048)
    spatial   = np.stack(frames_spa)   # (T, M, 4)
    T = roi_feats.shape[0]

    # I3D: extract per-frame (average of overlapping clips)
    i3d_feats = []
    clip_len = i3d.clip_len
    for t in range(T):
        start = max(0, t - clip_len // 2)
        end   = min(len(frames_i3d_raw), start + clip_len)
        clip  = frames_i3d_raw[start:end]
        i3d_feats.append(i3d.extract(clip))
    i3d_feats = np.stack(i3d_feats)   # (T, 1024)

    # Ground-truth boxes (filled from annotations if available)
    gt_subject = np.zeros((T, 4), dtype=np.float32)
    gt_object  = np.zeros((T, 4), dtype=np.float32)
    if gt_annotations is not None:
        _fill_gt_boxes(gt_subject, gt_object, gt_annotations, frame_step)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("roi_feats",  data=roi_feats,  compression="gzip")
        hf.create_dataset("spatial",    data=spatial,    compression="gzip")
        hf.create_dataset("i3d_feats",  data=i3d_feats,  compression="gzip")
        hf.create_dataset("gt_subject", data=gt_subject, compression="gzip")
        hf.create_dataset("gt_object",  data=gt_object,  compression="gzip")


def _fill_gt_boxes(gt_sub, gt_obj, ann, frame_step):
    """Populate GT arrays from annotation dict (dataset-specific)."""
    # Example for VidVRD: ann = {"trajectories": {tid: {fid: [x,y,w,h]}}}
    # This is a placeholder; adapt per dataset format.
    pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    choices=["vidvrd", "vidor"], required=True)
    parser.add_argument("--data_root",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--num_boxes",  type=int, default=30)
    parser.add_argument("--frame_step", type=int, default=4)
    args = parser.parse_args()

    frcnn = FasterRCNNExtractor(args.device, args.num_boxes)
    i3d   = I3DExtractor(args.device)

    video_dir = os.path.join(args.data_root, "videos")
    ann_dir   = os.path.join(args.data_root, "annotations")

    video_files = [f for f in os.listdir(video_dir)
                   if f.endswith((".mp4", ".avi", ".mkv"))]

    for vf in tqdm(video_files, desc="Extracting features"):
        video_id   = os.path.splitext(vf)[0]
        video_path = os.path.join(video_dir, vf)
        out_path   = os.path.join(args.output_dir, f"{video_id}.h5")

        if os.path.exists(out_path):
            continue

        ann_path = os.path.join(ann_dir, f"{video_id}.json")
        ann = None
        if os.path.isfile(ann_path):
            with open(ann_path) as f:
                ann = json.load(f)

        extract_video(
            video_path, out_path, frcnn, i3d,
            gt_annotations=ann,
            num_boxes=args.num_boxes,
            frame_step=args.frame_step,
        )

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
