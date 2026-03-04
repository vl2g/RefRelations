"""
utils/metrics.py

Evaluation metrics for few-shot referring relationships in videos.

Implements:
  • Asub_s−t  : spatiotemporal subject accuracy
  • Aobj_s−t  : spatiotemporal object  accuracy
  • Ar_s−t    : spatiotemporal relation accuracy
  • Asub_s    : spatial subject accuracy (avg IoU ≥ 0.5)
  • Aobj_s    : spatial object  accuracy
  • mIoU_sub  : mean IoU for subject
  • mIoU_obj  : mean IoU for object

Definitions follow Section 4.2 of the paper.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import numpy as np


def compute_iou(
    box1: np.ndarray,   # [x1, y1, x2, y2]
    box2: np.ndarray,   # [x1, y1, x2, y2]
) -> float:
    """Axis-aligned IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / max(union, 1e-6)


def trajectory_iou_sequence(
    pred_boxes: List[Optional[np.ndarray]],   # T boxes (None = skip)
    gt_boxes:   List[np.ndarray],             # T boxes
) -> List[float]:
    """Return per-frame IoU values; skipped frames receive 0."""
    ious = []
    for pred, gt in zip(pred_boxes, gt_boxes):
        if pred is None:
            ious.append(0.0)
        else:
            ious.append(compute_iou(pred, gt))
    return ious


def is_trajectory_correct_spatiotemporal(
    pred_boxes: List[Optional[np.ndarray]],
    gt_boxes:   List[np.ndarray],
    iou_thresh: float = 0.5,
    frame_ratio: float = 0.5,
) -> bool:
    """
    Spatiotemporal accuracy criterion (paper Sec. 4.2):
    A trajectory is correct if ≥ 50% of frames have IoU ≥ 0.5 with GT.
    """
    ious = trajectory_iou_sequence(pred_boxes, gt_boxes)
    if len(ious) == 0:
        return False
    correct = sum(1 for iou in ious if iou >= iou_thresh)
    return (correct / len(ious)) >= frame_ratio


def is_trajectory_correct_spatial(
    pred_boxes: List[Optional[np.ndarray]],
    gt_boxes:   List[np.ndarray],
    iou_thresh: float = 0.5,
) -> bool:
    """
    Spatial accuracy criterion: mean IoU across frames ≥ 0.5.
    """
    ious = trajectory_iou_sequence(pred_boxes, gt_boxes)
    return np.mean(ious) >= iou_thresh if ious else False


def mean_iou(
    pred_boxes: List[Optional[np.ndarray]],
    gt_boxes:   List[np.ndarray],
) -> float:
    """Average per-frame IoU over the trajectory."""
    ious = trajectory_iou_sequence(pred_boxes, gt_boxes)
    return float(np.mean(ious)) if ious else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Batch evaluation
# ─────────────────────────────────────────────────────────────────────────────

class MetricAccumulator:
    """
    Accumulate per-episode predictions and compute all 7 metrics at the end.
    """

    def __init__(
        self,
        iou_thresh: float  = 0.5,
        frame_ratio: float = 0.5,
    ) -> None:
        self.iou_thresh  = iou_thresh
        self.frame_ratio = frame_ratio
        self._reset()

    def _reset(self) -> None:
        self.sub_st_correct: List[bool] = []
        self.obj_st_correct: List[bool] = []
        self.sub_s_correct:  List[bool] = []
        self.obj_s_correct:  List[bool] = []
        self.sub_miou:       List[float] = []
        self.obj_miou:       List[float] = []

    def update(
        self,
        pred_sub_boxes: List[Optional[np.ndarray]],   # T boxes for subject
        pred_obj_boxes: List[Optional[np.ndarray]],   # T boxes for object
        gt_sub_boxes:   List[np.ndarray],
        gt_obj_boxes:   List[np.ndarray],
    ) -> None:
        self.sub_st_correct.append(
            is_trajectory_correct_spatiotemporal(
                pred_sub_boxes, gt_sub_boxes, self.iou_thresh, self.frame_ratio
            )
        )
        self.obj_st_correct.append(
            is_trajectory_correct_spatiotemporal(
                pred_obj_boxes, gt_obj_boxes, self.iou_thresh, self.frame_ratio
            )
        )
        self.sub_s_correct.append(
            is_trajectory_correct_spatial(
                pred_sub_boxes, gt_sub_boxes, self.iou_thresh
            )
        )
        self.obj_s_correct.append(
            is_trajectory_correct_spatial(
                pred_obj_boxes, gt_obj_boxes, self.iou_thresh
            )
        )
        self.sub_miou.append(mean_iou(pred_sub_boxes, gt_sub_boxes))
        self.obj_miou.append(mean_iou(pred_obj_boxes, gt_obj_boxes))

    def compute(self) -> Dict[str, float]:
        n = len(self.sub_st_correct)
        if n == 0:
            return {}

        asub_st = 100.0 * np.mean(self.sub_st_correct)
        aobj_st = 100.0 * np.mean(self.obj_st_correct)
        # Relation correct only if both subject AND object trajectories correct (spatiotemporal)
        ar_st   = 100.0 * np.mean([
            s and o for s, o in zip(self.sub_st_correct, self.obj_st_correct)
        ])
        asub_s  = 100.0 * np.mean(self.sub_s_correct)
        aobj_s  = 100.0 * np.mean(self.obj_s_correct)
        miou_sub = 100.0 * np.mean(self.sub_miou)
        miou_obj = 100.0 * np.mean(self.obj_miou)

        return {
            "Asub_s-t":  round(asub_st, 1),
            "Asub_s":    round(asub_s,  1),
            "mIoU_sub":  round(miou_sub, 1),
            "Aobj_s-t":  round(aobj_st, 1),
            "Aobj_s":    round(aobj_s,  1),
            "mIoU_obj":  round(miou_obj, 1),
            "Ar_s-t":    round(ar_st,   1),
            "N_episodes": n,
        }

    def reset(self) -> None:
        self._reset()
