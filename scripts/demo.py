"""
scripts/demo.py  –  Run few-shot referring relationship on a custom video.

Usage:
    python scripts/demo.py \\
        --video  path/to/test.mp4 \\
        --query  "plane,fly above,person" \\
        --support_dir  path/to/support_videos/ \\
        --checkpoint   checkpoints/best.pth \\
        --config       configs/default.yaml \\
        --output_video result.mp4

The support_dir should contain short video clips sharing the query predicate.
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

# Allow import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.full_model import FewShotReferringRelationship
from scripts.extract_features import FasterRCNNExtractor, I3DExtractor
from train import build_model, load_config


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_video_to_dict(
    video_path: str,
    frcnn: FasterRCNNExtractor,
    i3d: I3DExtractor,
    num_boxes: int = 30,
    frame_step: int = 4,
    max_frames: int = 30,
    device: torch.device = torch.device("cpu"),
) -> Tuple[dict, List[np.ndarray]]:
    """
    Extract features from a video and return a feature dict + raw frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    roi_list, spa_list, i3d_list, raw_frames = [], [], [], []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % frame_step == 0 and len(roi_list) < max_frames:
            roi, spa = frcnn.extract(frame)
            roi_list.append(roi)
            spa_list.append(spa)
            raw_frames.append(frame)
        fi += 1
    cap.release()

    if not roi_list:
        raise ValueError(f"No frames extracted from {video_path}")

    T = len(roi_list)
    # I3D (per-frame clips)
    for t in range(T):
        cl = raw_frames[max(0, t - 8):t + 8]
        i3d_list.append(i3d.extract(cl))

    roi_feats = torch.tensor(np.stack(roi_list), dtype=torch.float32).to(device)  # (T,M,2048)
    spatial   = torch.tensor(np.stack(spa_list), dtype=torch.float32).to(device)  # (T,M,4)
    i3d_feats = torch.tensor(np.stack(i3d_list), dtype=torch.float32).to(device)  # (T,1024)

    feats = {
        "roi_feats":  roi_feats,
        "spatial":    spatial,
        "i3d_feats":  i3d_feats,
        "gt_subject": torch.zeros(T, 4, device=device),
        "gt_object":  torch.zeros(T, 4, device=device),
    }
    return feats, raw_frames


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def draw_trajectory(
    frames: List[np.ndarray],
    subj_traj: List[int],
    obj_traj:  List[int],
    spatial: torch.Tensor,     # (T, M, 4) normalised
    subject_name: str,
    object_name:  str,
) -> List[np.ndarray]:
    out_frames = []
    for t, frame in enumerate(frames):
        vis = frame.copy()
        H, W = frame.shape[:2]
        # Subject – red
        j = subj_traj[t]
        if j >= 0:
            box = spatial[t, j].cpu().numpy()
            x1, y1, x2, y2 = (box * [W, H, W, H]).astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, subject_name, (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        # Object – blue
        k = obj_traj[t]
        if k >= 0:
            box = spatial[t, k].cpu().numpy()
            x1, y1, x2, y2 = (box * [W, H, W, H]).astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis, object_name, (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        out_frames.append(vis)
    return out_frames


def save_video(frames: List[np.ndarray], path: str, fps: int = 10) -> None:
    if not frames:
        return
    H, W = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"[Demo] Output video saved to: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",        required=True)
    parser.add_argument("--query",        required=True,
                        help="'subject,predicate,object' e.g. 'plane,fly above,person'")
    parser.add_argument("--support_dir",  required=True)
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--config",       default="configs/default.yaml")
    parser.add_argument("--output_video", default="result.mp4")
    parser.add_argument("--solver",       default="belief_propagation")
    parser.add_argument("--device",       default="cuda")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Parse query
    parts = [p.strip() for p in args.query.split(",")]
    if len(parts) != 3:
        raise ValueError("--query must be 'subject,predicate,object'")
    subj_name, pred_name, obj_name = parts

    # Load GloVe (minimal inline version for demo)
    from datasets.base_dataset import VideoRelationDataset
    glove = VideoRelationDataset._load_glove(cfg["data"]["glove_path"])
    def wemb(word):
        v = glove.get(word.lower(), np.zeros(300, dtype=np.float32))
        return torch.tensor(v, device=device)

    query = {
        "subject": subj_name, "predicate": pred_name, "object": obj_name,
        "subject_emb": wemb(subj_name),
        "object_emb":  wemb(obj_name),
        "pred_emb":    wemb(pred_name),
    }

    # Feature extractors
    frcnn = FasterRCNNExtractor(device=args.device, num_boxes=cfg["model"]["num_boxes_per_frame"])
    i3d   = I3DExtractor(device=args.device)

    # Extract test video
    print("[Demo] Extracting test video features …")
    test_feats, test_frames = extract_video_to_dict(
        args.video, frcnn, i3d, device=device,
        num_boxes=cfg["model"]["num_boxes_per_frame"],
        max_frames=30,
    )

    # Extract support videos
    support_dir = args.support_dir
    support_vids = [
        os.path.join(support_dir, f)
        for f in os.listdir(support_dir)
        if f.endswith((".mp4", ".avi", ".mkv"))
    ][:cfg["eval"]["support_size"]]

    print(f"[Demo] Found {len(support_vids)} support video(s).")
    support_feats = []
    for sv in support_vids:
        sf, _ = extract_video_to_dict(sv, frcnn, i3d, device=device, max_frames=30)
        support_feats.append(sf)

    # Load model
    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Predict
    print("[Demo] Running inference …")
    subj_traj, obj_traj = model.predict(test_feats, support_feats, query, solver=args.solver)
    print(f"[Demo] Subject trajectory: {subj_traj}")
    print(f"[Demo] Object  trajectory: {obj_traj}")

    # Visualise
    vis_frames = draw_trajectory(
        test_frames, subj_traj, obj_traj,
        test_feats["spatial"], subj_name, obj_name,
    )
    save_video(vis_frames, args.output_video)


if __name__ == "__main__":
    main()
