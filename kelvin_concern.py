"""
Compute ACES score statistics for SAM candidate masks.

This script is a non-Streamlit batch version of the manual tuning code.
It does not compute IoU. It computes score distributions at two levels:

1. Sequence level:
   - mean of frame-wise best candidate scores
   - mean score over all candidate masks in the sequence
   - variance of scores over all candidate masks in the sequence

2. Frame level:
   - max score among candidate masks
   - min score among candidate masks
   - mean score among candidate masks
   - variance of candidate-mask scores

Expected dataset structure under --root:

SAM_results/
  breakdance/
    gt_json/
      00000_gt.json
    json/
      00000_result.json
    masks/                    # not required by this script
    selected_by_sam/           # not required by this script

The result json is expected to contain:
  data["annotations"][i]["segmentation"] as COCO-style RLE.

Example:
  python compute_score_statistics.py ^
      --root "D:\\uwb thesis\\RelatedData\\DAVIS-data\\DAVIS\\SAM_results" ^
      --dataset breakdance ^
      --output-dir "D:\\uwb thesis\\score_statistics"

Use --dataset with multiple names, or omit --dataset to process all sequences.
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    from pycocotools import mask as mask_utils
except ImportError:  # compressed RLE requires pycocotools
    mask_utils = None


DEFAULT_ROOT = r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results"
EPS = 1e-8


# =========================================================
# RLE / mask utilities
# =========================================================

def decode_rle(segmentation_rle: Dict[str, Any]) -> np.ndarray:
    """
    Decode COCO-style RLE from JSON.

    Supports:
      1. compressed RLE: counts is str/bytes
      2. uncompressed RLE: counts is list

    Returns:
      np.ndarray[bool], shape = (H, W)
    """
    if segmentation_rle is None:
        raise ValueError("segmentation_rle is None.")

    size = segmentation_rle.get("size", None)
    counts = segmentation_rle.get("counts", None)

    if size is None or counts is None:
        raise ValueError("Invalid RLE: missing size or counts.")

    h, w = int(size[0]), int(size[1])

    if isinstance(counts, str):
        if mask_utils is None:
            raise ImportError(
                "pycocotools is required to decode compressed RLE counts. "
                "Install with: pip install pycocotools"
            )
        rle = {"size": [h, w], "counts": counts.encode("utf-8")}
        return mask_utils.decode(rle).astype(bool)

    if isinstance(counts, bytes):
        if mask_utils is None:
            raise ImportError(
                "pycocotools is required to decode compressed RLE counts. "
                "Install with: pip install pycocotools"
            )
        rle = {"size": [h, w], "counts": counts}
        return mask_utils.decode(rle).astype(bool)

    if isinstance(counts, list):
        flat: List[int] = []
        value = 0
        for run_len in counts:
            flat.extend([value] * int(run_len))
            value = 1 - value

        arr = np.array(flat, dtype=np.uint8)
        expected = h * w
        if arr.size < expected:
            arr = np.pad(arr, (0, expected - arr.size), mode="constant")
        elif arr.size > expected:
            arr = arr[:expected]

        # COCO RLE is column-major.
        mask = arr.reshape((w, h)).T
        return mask.astype(bool)

    raise TypeError(f"Unsupported RLE counts type: {type(counts)}")


# =========================================================
# ACES feature and score functions
#   Feature vector = [A, C, E, Sil]
# =========================================================

def compute_distance_transform(h: int, w: int) -> np.ndarray:
    border_mask = np.zeros((h, w), np.uint8)
    border_mask[1:-1, 1:-1] = 1
    return cv2.distanceTransform(
        border_mask,
        distanceType=cv2.DIST_L2,
        maskSize=5,
    )


def area_term_parabola(x: float, d: float, eps: float = 1e-6) -> float:
    d = float(np.clip(d, eps, 1.0))
    x = float(np.clip(x, 0.0, 1.0))
    val = 1.0 - ((x - d) / d) ** 2
    return float(np.clip(val, 0.0, 1.0))


def compute_silhouette_score_v2(mask: np.ndarray) -> float:
    mask_u8 = (mask.astype(np.uint8) > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return 0.0

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    if area <= 0:
        return 0.0

    hull = cv2.convexHull(largest)
    hull_area = float(cv2.contourArea(hull))
    solidity = 0.0 if hull_area <= 0 else float(np.clip(area / hull_area, 0.0, 1.0))

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        fragmentation = 0.0
    else:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        total_area = float(np.sum(areas))
        if total_area <= 0:
            fragmentation = 0.0
        else:
            thr = 0.01 * total_area
            large_areas = areas[areas >= thr]
            if large_areas.size <= 1:
                fragmentation = 0.0
            else:
                largest_area = float(np.max(large_areas))
                fragmentation = 1.0 - (largest_area / float(np.sum(large_areas)))

    fragmentation = float(np.clip(fragmentation, 0.0, 1.0))

    bg = (1 - mask_u8).astype(np.uint8)
    bg_pad = np.pad(bg, pad_width=1, mode="constant", constant_values=1)
    h, w = bg_pad.shape
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    bg_ff = bg_pad.copy()
    cv2.floodFill(bg_ff, ff_mask, (0, 0), 0)

    holes_map = bg_ff[1:-1, 1:-1] == 1
    holes_area = float(np.count_nonzero(holes_map))
    hole_ratio = float(np.clip(holes_area / area, 0.0, 1.0)) if area > 0 else 0.0

    sil = solidity * (1.0 - fragmentation) * (1.0 - hole_ratio)
    return float(np.clip(sil, 0.0, 1.0))


def normalize_weights(w: Sequence[float]) -> np.ndarray:
    w_arr = np.asarray(w, dtype=float)
    w_arr = np.clip(w_arr, 0.0, None)
    s = float(w_arr.sum())
    if s <= EPS:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    return w_arr / s


def mask_features(
    mask_bool: np.ndarray,
    W: int,
    H: int,
    dist_map: np.ndarray,
    d_max: float,
    q_border: float,
    t_area: float,
) -> np.ndarray:
    img_area = W * H
    cx, cy = W / 2, H / 2

    area_px = int(mask_bool.sum())
    if area_px <= 0:
        return np.array([0, 0, 0, 0], dtype=float)

    # A: area prior
    A_raw = area_px / img_area
    A = area_term_parabola(A_raw, t_area)

    # C: center prior
    ys, xs = np.where(mask_bool)
    mx, my = xs.mean(), ys.mean()
    Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
    C = 1.0 - float(np.clip(Dp, 0.0, 1.0))

    # E: border-distance / interior support term
    q = float(np.quantile(dist_map[mask_bool], q_border))
    E = float(np.clip(q / d_max, 0.0, 1.0))

    # Sil: silhouette structure
    Sil = compute_silhouette_score_v2(mask_bool)

    return np.array([A, C, E, Sil], dtype=float)


def compute_score(features: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(features, weights))


# =========================================================
# Dataset loading
# =========================================================

def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_datasets(root: Path) -> List[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def list_frame_ids(dataset_dir: Path) -> List[str]:
    """
    Prefer gt_json because it was the frame source in the tuning script.
    Fall back to result json files if gt_json is unavailable.
    """
    frame_ids: List[str] = []

    gt_dir = dataset_dir / "gt_json"
    if gt_dir.exists():
        for p in gt_dir.glob("*_gt.json"):
            m = re.match(r"(\d+)_gt\.json$", p.name)
            if m:
                frame_ids.append(m.group(1))

    if frame_ids:
        return sorted(frame_ids)

    result_dir = dataset_dir / "json"
    if result_dir.exists():
        for p in result_dir.glob("*_result.json"):
            m = re.match(r"(\d+)_result\.json$", p.name)
            if m:
                frame_ids.append(m.group(1))

    return sorted(frame_ids)


def load_candidate_masks_from_json(result_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Decode all candidate masks from result json.

    Expected annotation format:
      ann["id"]
      ann["box_id"]
      ann["rank_in_box"]
      ann["sam_score"]
      ann["segmentation"]
    """
    anns = result_json.get("annotations", [])
    candidates: List[Dict[str, Any]] = []

    for idx, ann in enumerate(anns):
        ann_id = int(ann.get("id", idx))

        # Main expected key is "segmentation". The fallback makes the script
        # slightly more tolerant of related JSON exports.
        segmentation = ann.get("segmentation", ann.get("segmentation_rle", None))
        if segmentation is None:
            raise KeyError(f"Annotation id={ann_id} has no segmentation RLE.")

        mask = decode_rle(segmentation)

        candidates.append({
            "id": ann_id,
            "box_id": int(ann.get("box_id", -1)),
            "rank_in_box": int(ann.get("rank_in_box", -1)),
            "sam_score": float(ann.get("sam_score", np.nan)),
            "mask": mask,
        })

    return candidates


def frame_shape_from_gt_or_candidates(
    dataset_dir: Path,
    frame_id: str,
    candidates: List[Dict[str, Any]],
) -> Tuple[int, int, str]:
    """
    Return (H, W, shape_source).
    Prefer gt_json width/height. Fall back to the first candidate mask shape.
    """
    gt_json_path = dataset_dir / "gt_json" / f"{frame_id}_gt.json"

    if gt_json_path.exists():
        gt_data = read_json(gt_json_path)
        if "height" in gt_data and "width" in gt_data:
            return int(gt_data["height"]), int(gt_data["width"]), "gt_json_height_width"
        if "segmentation_rle" in gt_data:
            gt_mask = decode_rle(gt_data["segmentation_rle"])
            return int(gt_mask.shape[0]), int(gt_mask.shape[1]), "gt_json_rle_shape"

    if candidates:
        mask = candidates[0]["mask"]
        return int(mask.shape[0]), int(mask.shape[1]), "first_candidate_shape"

    raise ValueError("Cannot infer frame shape: no GT json and no candidate masks.")


# =========================================================
# Statistics
# =========================================================

def safe_mean(values: Sequence[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def safe_var(values: Sequence[float], ddof: int = 0) -> float:
    if len(values) == 0:
        return float("nan")
    if len(values) <= ddof:
        return float("nan")
    return float(np.var(values, ddof=ddof))


def evaluate_frame(
    dataset_name: str,
    dataset_dir: Path,
    frame_id: str,
    weights: np.ndarray,
    t_area: float,
    q_border: float,
    variance_ddof: int,
    keep_candidate_details: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[float]]:
    result_json_path = dataset_dir / "json" / f"{frame_id}_result.json"

    if not result_json_path.exists():
        return {
            "dataset": dataset_name,
            "frame_id": frame_id,
            "num_candidates_total": 0,
            "num_candidates_scored": 0,
            "best_candidate_id": np.nan,
            "frame_max_score": np.nan,
            "frame_min_score": np.nan,
            "frame_mean_score": np.nan,
            "frame_score_variance": np.nan,
            "frame_score_range": np.nan,
            "top_score_minus_mean": np.nan,
            "note": "missing result_json",
        }, [], []

    candidate_details: List[Dict[str, Any]] = []
    scored_scores: List[float] = []

    try:
        result_json = read_json(result_json_path)
        candidates = load_candidate_masks_from_json(result_json)

        if len(candidates) == 0:
            return {
                "dataset": dataset_name,
                "frame_id": frame_id,
                "num_candidates_total": 0,
                "num_candidates_scored": 0,
                "best_candidate_id": np.nan,
                "frame_max_score": np.nan,
                "frame_min_score": np.nan,
                "frame_mean_score": np.nan,
                "frame_score_variance": np.nan,
                "frame_score_range": np.nan,
                "top_score_minus_mean": np.nan,
                "note": "no candidate masks",
            }, [], []

        H, W, shape_source = frame_shape_from_gt_or_candidates(dataset_dir, frame_id, candidates)
        dist_map = compute_distance_transform(H, W)
        d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

        best_candidate_id: Optional[int] = None
        best_score = -np.inf
        shape_mismatch_count = 0

        for cand in candidates:
            mask = cand["mask"]
            if mask.shape != (H, W):
                shape_mismatch_count += 1
                continue

            feats = mask_features(
                mask_bool=mask,
                W=W,
                H=H,
                dist_map=dist_map,
                d_max=d_max,
                q_border=q_border,
                t_area=t_area,
            )
            score = compute_score(feats, weights)
            weighted_terms = feats * weights

            scored_scores.append(score)

            if score > best_score:
                best_score = score
                best_candidate_id = int(cand["id"])

            if keep_candidate_details:
                candidate_details.append({
                    "dataset": dataset_name,
                    "frame_id": frame_id,
                    "candidate_id": int(cand["id"]),
                    "box_id": int(cand["box_id"]),
                    "rank_in_box": int(cand["rank_in_box"]),
                    "sam_score": cand["sam_score"],
                    "score": score,
                    "A": float(feats[0]),
                    "C": float(feats[1]),
                    "E": float(feats[2]),
                    "Sil": float(feats[3]),
                    "wA_A": float(weighted_terms[0]),
                    "wC_C": float(weighted_terms[1]),
                    "wE_E": float(weighted_terms[2]),
                    "wSil_Sil": float(weighted_terms[3]),
                })

        if len(scored_scores) == 0:
            return {
                "dataset": dataset_name,
                "frame_id": frame_id,
                "num_candidates_total": len(candidates),
                "num_candidates_scored": 0,
                "best_candidate_id": np.nan,
                "frame_max_score": np.nan,
                "frame_min_score": np.nan,
                "frame_mean_score": np.nan,
                "frame_score_variance": np.nan,
                "frame_score_range": np.nan,
                "top_score_minus_mean": np.nan,
                "note": f"no valid candidate after shape check; shape_source={shape_source}; mismatched={shape_mismatch_count}",
            }, candidate_details, []

        max_score = float(np.max(scored_scores))
        min_score = float(np.min(scored_scores))
        mean_score = float(np.mean(scored_scores))
        var_score = safe_var(scored_scores, ddof=variance_ddof)

        note = "ok"
        if shape_mismatch_count > 0:
            note = f"ok; skipped_shape_mismatch={shape_mismatch_count}; shape_source={shape_source}"

        frame_record = {
            "dataset": dataset_name,
            "frame_id": frame_id,
            "num_candidates_total": len(candidates),
            "num_candidates_scored": len(scored_scores),
            "best_candidate_id": best_candidate_id,
            "frame_max_score": max_score,
            "frame_min_score": min_score,
            "frame_mean_score": mean_score,
            "frame_score_variance": var_score,
            "frame_score_range": float(max_score - min_score),
            "top_score_minus_mean": float(max_score - mean_score),
            "note": note,
        }

        return frame_record, candidate_details, scored_scores

    except Exception as e:
        return {
            "dataset": dataset_name,
            "frame_id": frame_id,
            "num_candidates_total": 0,
            "num_candidates_scored": 0,
            "best_candidate_id": np.nan,
            "frame_max_score": np.nan,
            "frame_min_score": np.nan,
            "frame_mean_score": np.nan,
            "frame_score_variance": np.nan,
            "frame_score_range": np.nan,
            "top_score_minus_mean": np.nan,
            "note": f"error: {type(e).__name__}: {e}",
        }, [], []


def evaluate_sequence(
    dataset_name: str,
    dataset_dir: Path,
    weights: np.ndarray,
    raw_weights: Sequence[float],
    t_area: float,
    q_border: float,
    variance_ddof: int,
    keep_candidate_details: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    frame_ids = list_frame_ids(dataset_dir)

    frame_records: List[Dict[str, Any]] = []
    candidate_records: List[Dict[str, Any]] = []
    all_candidate_scores: List[float] = []

    for idx, frame_id in enumerate(frame_ids, start=1):
        print(f"[{dataset_name}] frame {frame_id} ({idx}/{len(frame_ids)})")
        frame_record, cand_records, scores = evaluate_frame(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            frame_id=frame_id,
            weights=weights,
            t_area=t_area,
            q_border=q_border,
            variance_ddof=variance_ddof,
            keep_candidate_details=keep_candidate_details,
        )
        frame_records.append(frame_record)
        candidate_records.extend(cand_records)
        all_candidate_scores.extend(scores)

    valid_best_scores = [
        float(r["frame_max_score"])
        for r in frame_records
        if pd.notna(r["frame_max_score"])
    ]

    valid_frame_ranges = [
        float(r["frame_score_range"])
        for r in frame_records
        if pd.notna(r["frame_score_range"])
    ]

    valid_top_minus_mean = [
        float(r["top_score_minus_mean"])
        for r in frame_records
        if pd.notna(r["top_score_minus_mean"])
    ]

    sequence_record = {
        "dataset": dataset_name,
        "num_frames_detected": len(frame_ids),
        "num_frames_scored": len(valid_best_scores),
        "total_candidates_scored": len(all_candidate_scores),

        # Required sequence-level statistics
        "mean_of_frame_best_scores": safe_mean(valid_best_scores),
        "all_candidate_mean_score": safe_mean(all_candidate_scores),
        "all_candidate_score_variance": safe_var(all_candidate_scores, ddof=variance_ddof),

        # Extra diagnostics useful for judging whether the best score is meaningfully separated
        "all_candidate_min_score": float(np.min(all_candidate_scores)) if all_candidate_scores else np.nan,
        "all_candidate_max_score": float(np.max(all_candidate_scores)) if all_candidate_scores else np.nan,
        "all_candidate_score_range": (
            float(np.max(all_candidate_scores) - np.min(all_candidate_scores))
            if all_candidate_scores else np.nan
        ),
        "mean_frame_score_range": safe_mean(valid_frame_ranges),
        "mean_top_score_minus_frame_mean": safe_mean(valid_top_minus_mean),

        # Parameter record
        "raw_wA": float(raw_weights[0]),
        "raw_wC": float(raw_weights[1]),
        "raw_wE": float(raw_weights[2]),
        "raw_wSil": float(raw_weights[3]),
        "norm_wA": float(weights[0]),
        "norm_wC": float(weights[1]),
        "norm_wE": float(weights[2]),
        "norm_wSil": float(weights[3]),
        "t_area": float(t_area),
        "q_border": float(q_border),
        "variance_ddof": int(variance_ddof),
        "note": "ok" if len(frame_ids) > 0 else "no frames found",
    }

    return sequence_record, frame_records, candidate_records


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute sequence-level and frame-level ACES score statistics."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=DEFAULT_ROOT,
        help="Root folder containing sequence folders, e.g., DAVIS/SAM_results.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="*",
        default=None,
        help="Sequence name(s) to process. Omit this argument to process all sequences under --root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Folder to save CSV outputs. Default: <root>/score_statistics_output.",
    )

    # Same defaults as the Streamlit tuning script.
    parser.add_argument("--wA", type=float, default=0.15, help="Raw weight for area term.")
    parser.add_argument("--wC", type=float, default=0.20, help="Raw weight for center term.")
    parser.add_argument("--wE", type=float, default=0.40, help="Raw weight for border/interior support term.")
    parser.add_argument("--wSil", type=float, default=0.25, help="Raw weight for silhouette term.")
    parser.add_argument("--t-area", type=float, default=0.20, help="Expected area ratio for A term.")
    parser.add_argument("--q-border", type=float, default=0.20, help="Quantile used by E term.")

    parser.add_argument(
        "--variance-ddof",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use 0 for population variance, 1 for sample variance. Default: 0.",
    )
    parser.add_argument(
        "--save-candidate-details",
        action="store_true",
        help="Also save one row per candidate mask. This can be large.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "score_statistics_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        dataset_names = args.dataset
    else:
        dataset_names = list_datasets(root)

    if not dataset_names:
        raise FileNotFoundError(f"No dataset folders found under root: {root}")

    raw_weights = [args.wA, args.wC, args.wE, args.wSil]
    weights = normalize_weights(raw_weights)

    print("Root:", root)
    print("Output:", output_dir)
    print("Datasets:", ", ".join(dataset_names))
    print(
        "Normalized weights:",
        f"A={weights[0]:.6f}, C={weights[1]:.6f}, E={weights[2]:.6f}, Sil={weights[3]:.6f}",
    )
    print(f"t_area={args.t_area}, q_border={args.q_border}, variance_ddof={args.variance_ddof}")

    sequence_records: List[Dict[str, Any]] = []
    frame_records_all: List[Dict[str, Any]] = []
    candidate_records_all: List[Dict[str, Any]] = []

    for dataset_name in dataset_names:
        dataset_dir = root / dataset_name
        if not dataset_dir.exists():
            print(f"WARNING: dataset folder does not exist: {dataset_dir}")
            sequence_records.append({
                "dataset": dataset_name,
                "num_frames_detected": 0,
                "num_frames_scored": 0,
                "total_candidates_scored": 0,
                "mean_of_frame_best_scores": np.nan,
                "all_candidate_mean_score": np.nan,
                "all_candidate_score_variance": np.nan,
                "all_candidate_min_score": np.nan,
                "all_candidate_max_score": np.nan,
                "all_candidate_score_range": np.nan,
                "mean_frame_score_range": np.nan,
                "mean_top_score_minus_frame_mean": np.nan,
                "raw_wA": float(raw_weights[0]),
                "raw_wC": float(raw_weights[1]),
                "raw_wE": float(raw_weights[2]),
                "raw_wSil": float(raw_weights[3]),
                "norm_wA": float(weights[0]),
                "norm_wC": float(weights[1]),
                "norm_wE": float(weights[2]),
                "norm_wSil": float(weights[3]),
                "t_area": float(args.t_area),
                "q_border": float(args.q_border),
                "variance_ddof": int(args.variance_ddof),
                "note": "dataset folder missing",
            })
            continue

        try:
            sequence_record, frame_records, candidate_records = evaluate_sequence(
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                weights=weights,
                raw_weights=raw_weights,
                t_area=args.t_area,
                q_border=args.q_border,
                variance_ddof=args.variance_ddof,
                keep_candidate_details=args.save_candidate_details,
            )
            sequence_records.append(sequence_record)
            frame_records_all.extend(frame_records)
            candidate_records_all.extend(candidate_records)
        except Exception as e:
            print(f"ERROR while processing dataset {dataset_name}: {e}")
            print(traceback.format_exc())
            sequence_records.append({
                "dataset": dataset_name,
                "num_frames_detected": 0,
                "num_frames_scored": 0,
                "total_candidates_scored": 0,
                "mean_of_frame_best_scores": np.nan,
                "all_candidate_mean_score": np.nan,
                "all_candidate_score_variance": np.nan,
                "all_candidate_min_score": np.nan,
                "all_candidate_max_score": np.nan,
                "all_candidate_score_range": np.nan,
                "mean_frame_score_range": np.nan,
                "mean_top_score_minus_frame_mean": np.nan,
                "raw_wA": float(raw_weights[0]),
                "raw_wC": float(raw_weights[1]),
                "raw_wE": float(raw_weights[2]),
                "raw_wSil": float(raw_weights[3]),
                "norm_wA": float(weights[0]),
                "norm_wC": float(weights[1]),
                "norm_wE": float(weights[2]),
                "norm_wSil": float(weights[3]),
                "t_area": float(args.t_area),
                "q_border": float(args.q_border),
                "variance_ddof": int(args.variance_ddof),
                "note": f"error: {type(e).__name__}: {e}",
            })

    sequence_df = pd.DataFrame(sequence_records)
    frame_df = pd.DataFrame(frame_records_all)

    sequence_csv = output_dir / "sequence_score_summary.csv"
    frame_csv = output_dir / "frame_score_summary.csv"

    sequence_df.to_csv(sequence_csv, index=False, encoding="utf-8-sig")
    frame_df.to_csv(frame_csv, index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print(" ", sequence_csv)
    print(" ", frame_csv)

    if args.save_candidate_details:
        candidate_df = pd.DataFrame(candidate_records_all)
        candidate_csv = output_dir / "candidate_score_details.csv"
        candidate_df.to_csv(candidate_csv, index=False, encoding="utf-8-sig")
        print(" ", candidate_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
