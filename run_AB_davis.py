import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


EPS = 1e-8

# =========================================================
# Default sequence subset used in the current thesis analysis
# =========================================================

DEFAULT_SEQUENCES = [
    "mallard-water",
    "flamingo",
    "dance-twirl",
    "lucia",
    "drift-turn",
    "drift-chicane",
    "breakdance",
    "car-roundabout",
    "bmx-trees",
    "car-shadow",
    "drift-straight",
    "horsejump-high",
    "kite-walk",
    "libby",
    "hike",
    "hockey",
    "bus",
    "dance-jump",
    "kite-surf",
    "horsejump-low",
    "boat",
    "cows",
    "dog",
    "blackswan",
    "elephant",
    "breakdance-flare",
    "goat",
    "camel",
    "mallard-fly",
    "bear",
    "car-turn",
    "bmx-bumps",
    "dog-agility",
]


# =========================================================
# A default weights + B default params
# =========================================================

DEFAULT_A_WEIGHTS = {
    "wA": 0.15,
    "wC": 0.20,
    "wE": 0.40,
    "wS": 0.25,
}

DEFAULT_B_PARAMS = {
    "q_border": 0.25,
    "t_area": 0.25,
    "c_target": 1.0,
    "e_target": 0.0,
    "s_target": 1.0,
}


# =========================================================
# RLE / image / mask utilities
# =========================================================

def decode_rle(segmentation_rle: Dict[str, Any]) -> np.ndarray:
    """
    Decode COCO-style RLE from JSON.

    Supports:
      1. compressed RLE: counts is string / bytes
      2. uncompressed RLE: counts is list

    Returns:
      bool mask with shape (H, W)
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
            raise ImportError("pycocotools is required for compressed RLE decoding.")
        rle = {"size": [h, w], "counts": counts.encode("utf-8")}
        return mask_utils.decode(rle).astype(bool)

    if isinstance(counts, bytes):
        if mask_utils is None:
            raise ImportError("pycocotools is required for compressed RLE decoding.")
        rle = {"size": [h, w], "counts": counts}
        return mask_utils.decode(rle).astype(bool)

    if isinstance(counts, list):
        flat = []
        value = 0
        for run_len in counts:
            flat.extend([value] * int(run_len))
            value = 1 - value

        arr = np.asarray(flat, dtype=np.uint8)
        expected = h * w

        if arr.size < expected:
            arr = np.pad(arr, (0, expected - arr.size), mode="constant")
        elif arr.size > expected:
            arr = arr[:expected]

        # COCO RLE is column-major.
        return arr.reshape((w, h)).T.astype(bool)

    raise TypeError(f"Unsupported RLE counts type: {type(counts)}")


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union <= 0:
        return 0.0
    return float(inter / union)


def write_mask_png(path: Path, mask_bool: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask_bool.astype(np.uint8) * 255)


# =========================================================
# Feature functions copied from A/B scoring logic
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

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )

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

    holes_map = (bg_ff[1:-1, 1:-1] == 1)
    holes_area = float(np.count_nonzero(holes_map))
    hole_ratio = float(np.clip(holes_area / area, 0.0, 1.0)) if area > 0 else 0.0

    sil = solidity * (1.0 - fragmentation) * (1.0 - hole_ratio)
    return float(np.clip(sil, 0.0, 1.0))


def mask_features_A_observed(
    mask_bool: np.ndarray,
    W: int,
    H: int,
    dist_map: np.ndarray,
    d_max: float,
    q_border: float,
    t_area: float,
) -> np.ndarray:
    """
    Original A-version observed feature vector:
      [A, C, E_dist, Sil]
    """
    img_area = W * H
    cx, cy = W / 2.0, H / 2.0

    area_px = int(mask_bool.sum())
    if area_px <= 0:
        return np.zeros(4, dtype=float)

    # A: area compatibility with t_area.
    area_ratio = area_px / img_area
    A = area_term_parabola(area_ratio, t_area)

    # C: centeredness. 1=center, 0=far.
    ys, xs = np.where(mask_bool)
    mx, my = xs.mean(), ys.mean()
    Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
    C = 1.0 - float(np.clip(Dp, 0.0, 1.0))

    # E: border-distance / interior support. 1=far from border, 0=near border.
    if np.any(mask_bool):
        q = float(np.quantile(dist_map[mask_bool], q_border))
    else:
        q = 0.0
    E_dist = float(np.clip(q / d_max, 0.0, 1.0))

    # Silhouette.
    Sil = compute_silhouette_score_v2(mask_bool)

    return np.asarray([A, C, E_dist, Sil], dtype=float)


def mask_features_B_match(
    mask_bool: np.ndarray,
    W: int,
    H: int,
    dist_map: np.ndarray,
    d_max: float,
    params: Dict[str, float],
) -> np.ndarray:
    """
    B-version target-matching feature vector:
      [A_match, C_match, E_match, S_match]

    B assumptions:
      t_area:   target area ratio for A_match.
      c_target: expected centeredness.
      e_target: expected edge proximity.
      s_target: expected silhouette quality.

    Note:
      A_match uses the A area parabola directly.
      B's E target is edge proximity, while A's E observed feature is
      border-distance support. Therefore:
        edge_proximity = 1 - E_dist
        E_match = 1 - abs(e_target - edge_proximity)
    """
    q_border = float(np.clip(params.get("q_border", DEFAULT_B_PARAMS["q_border"]), 0.0, 1.0))
    t_area = float(np.clip(params.get("t_area", DEFAULT_B_PARAMS["t_area"]), 1e-6, 1.0))
    c_target = float(np.clip(params.get("c_target", DEFAULT_B_PARAMS["c_target"]), 0.0, 1.0))
    e_target = float(np.clip(params.get("e_target", DEFAULT_B_PARAMS["e_target"]), 0.0, 1.0))
    s_target = float(np.clip(params.get("s_target", DEFAULT_B_PARAMS["s_target"]), 0.0, 1.0))

    A_obs, C_obs, E_dist, Sil_obs = mask_features_A_observed(
        mask_bool=mask_bool,
        W=W,
        H=H,
        dist_map=dist_map,
        d_max=d_max,
        q_border=q_border,
        t_area=t_area,
    )

    edge_proximity = 1.0 - E_dist

    A_match = A_obs
    C_match = 1.0 - abs(c_target - C_obs)
    E_match = 1.0 - abs(e_target - edge_proximity)
    S_match = 1.0 - abs(s_target - Sil_obs)

    return np.asarray([
        np.clip(A_match, 0.0, 1.0),
        np.clip(C_match, 0.0, 1.0),
        np.clip(E_match, 0.0, 1.0),
        np.clip(S_match, 0.0, 1.0),
    ], dtype=float)


def normalize_weights(weights: Dict[str, float]) -> np.ndarray:
    w = np.asarray(
        [
            weights.get("wA", DEFAULT_A_WEIGHTS["wA"]),
            weights.get("wC", DEFAULT_A_WEIGHTS["wC"]),
            weights.get("wE", DEFAULT_A_WEIGHTS["wE"]),
            weights.get("wS", DEFAULT_A_WEIGHTS["wS"]),
        ],
        dtype=float,
    )
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= EPS:
        return np.asarray([0.25, 0.25, 0.25, 0.25], dtype=float)
    return w / s


# =========================================================
# Dataset loading
# =========================================================

def load_gt_from_gt_json(gt_json_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    gt_data = read_json(gt_json_path)
    gt_mask = decode_rle(gt_data["segmentation_rle"])
    return gt_mask, gt_data


def load_candidates_from_result_json(result_json_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    data = read_json(result_json_path)
    anns = data.get("annotations", [])

    candidates = []
    raw_scores = data.get("raw_scores", None)

    for idx, ann in enumerate(anns):
        ann_id = int(ann.get("id", idx))
        mask = decode_rle(ann["segmentation"])

        if raw_scores is not None and idx < len(raw_scores):
            sam_score = float(raw_scores[idx])
        else:
            sam_score = float(ann.get("sam_score", ann.get("score", 0.0)))

        candidates.append({
            "idx": idx,
            "id": ann_id,
            "box_id": int(ann.get("box_id", -1)),
            "rank_in_box": int(ann.get("rank_in_box", -1)),
            "sam_score": sam_score,
            "mask": mask,
        })

    return candidates, data


def list_available_sequences(data_root: Path) -> List[str]:
    seqs = [
        p.name for p in data_root.iterdir()
        if p.is_dir() and (p / "json").exists() and (p / "gt_json").exists()
    ]
    return sorted(seqs)


def load_optimized_b_params(b_root: Path, sequence: str) -> Optional[Dict[str, float]]:
    path = b_root / sequence / "optimized_params_B.json"
    if not path.exists():
        return None

    obj = read_json(path)
    if obj.get("skipped", False):
        return None

    opt = obj.get("optimized", None)
    if not isinstance(opt, dict):
        return None

    return {
        "q_border": float(opt.get("q_border", DEFAULT_B_PARAMS["q_border"])),
        "t_area": float(opt.get("t_area", DEFAULT_B_PARAMS["t_area"])),
        "c_target": float(opt.get("c_target", DEFAULT_B_PARAMS["c_target"])),
        "e_target": float(opt.get("e_target", DEFAULT_B_PARAMS["e_target"])),
        "s_target": float(opt.get("s_target", DEFAULT_B_PARAMS["s_target"])),
    }


# =========================================================
# Evaluation
# =========================================================

def summarize_array(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": np.nan, "min": np.nan, "max": np.nan, "std": np.nan, "n": 0}
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std(ddof=0)),
        "n": int(arr.size),
    }


def evaluate_sequence_AB(
    sequence: str,
    data_root: Path,
    b_root: Path,
    out_root: Path,
    weights_vec: np.ndarray,
    save_selected_png: bool = False,
    force: bool = False,
    save_candidate_details: bool = False,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    """
    Evaluate AB version for one sequence.

    AB version:
      - Uses B optimized assumptions per sequence:
          t_area, c_target, e_target, s_target, q_border
      - Uses A default weights:
          score_AB = wA*A_match + wC*C_match + wE*E_match + wS*S_match

    Also computes:
      - SAM raw-score choice
      - B unweighted choice:
          score_B = A_match + C_match + E_match + S_match
      - oracle choice:
          highest IoU candidate
    """
    seq_data_dir = data_root / sequence
    seq_b_dir = b_root / sequence
    seq_out_dir = out_root / sequence
    seq_out_dir.mkdir(parents=True, exist_ok=True)

    params = load_optimized_b_params(b_root, sequence)
    if params is None:
        return None, [], [], "missing_or_invalid_optimized_params_B.json"

    json_dir = seq_data_dir / "json"
    gt_dir = seq_data_dir / "gt_json"

    if not json_dir.exists() or not gt_dir.exists():
        return None, [], [], "missing_json_or_gt_json_dir"

    result_files = sorted(json_dir.glob("*_result.json"))
    if not result_files:
        return None, [], [], "no_result_json_files"

    per_frame_rows = []
    candidate_rows = []

    for result_path in result_files:
        key = result_path.stem.replace("_result", "")
        gt_path = gt_dir / f"{key}_gt.json"

        if not gt_path.exists():
            continue

        try:
            gt_mask, _ = load_gt_from_gt_json(gt_path)
            candidates, result_obj = load_candidates_from_result_json(result_path)
        except Exception as e:
            per_frame_rows.append({
                "sequence": sequence,
                "frame": key,
                "note": f"load_error: {e}",
            })
            continue

        if not candidates:
            per_frame_rows.append({
                "sequence": sequence,
                "frame": key,
                "note": "no_candidates",
            })
            continue

        H, W = gt_mask.shape
        dist_map = compute_distance_transform(H, W)
        d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

        rows = []
        for cand in candidates:
            mask = cand["mask"]
            if mask.shape != gt_mask.shape:
                continue

            terms = mask_features_B_match(
                mask_bool=mask,
                W=W,
                H=H,
                dist_map=dist_map,
                d_max=d_max,
                params=params,
            )
            A_match, C_match, E_match, S_match = terms.tolist()

            score_B = float(np.sum(terms))
            weighted_terms = terms * weights_vec
            score_AB = float(np.dot(terms, weights_vec))
            iou = compute_iou(mask, gt_mask)

            row = {
                "sequence": sequence,
                "frame": key,
                "idx": int(cand["idx"]),
                "id": int(cand["id"]),
                "box_id": int(cand["box_id"]),
                "rank_in_box": int(cand["rank_in_box"]),
                "sam_score": float(cand["sam_score"]),
                "IoU": float(iou),

                "score_B_unweighted": score_B,
                "score_AB": score_AB,

                "A_match": float(A_match),
                "C_match": float(C_match),
                "E_match": float(E_match),
                "S_match": float(S_match),

                "wA_A_match": float(weighted_terms[0]),
                "wC_C_match": float(weighted_terms[1]),
                "wE_E_match": float(weighted_terms[2]),
                "wS_S_match": float(weighted_terms[3]),

                "AB_A_share": float(weighted_terms[0] / score_AB) if score_AB > EPS else np.nan,
                "AB_C_share": float(weighted_terms[1] / score_AB) if score_AB > EPS else np.nan,
                "AB_E_share": float(weighted_terms[2] / score_AB) if score_AB > EPS else np.nan,
                "AB_S_share": float(weighted_terms[3] / score_AB) if score_AB > EPS else np.nan,

                "mask": mask,
            }
            rows.append(row)

            if save_candidate_details:
                candidate_rows.append({k: v for k, v in row.items() if k != "mask"})

        if not rows:
            per_frame_rows.append({
                "sequence": sequence,
                "frame": key,
                "note": "no_valid_candidates",
            })
            continue

        sam_choice = max(rows, key=lambda x: x["sam_score"])
        b_choice = max(rows, key=lambda x: x["score_B_unweighted"])
        ab_choice = max(rows, key=lambda x: x["score_AB"])
        oracle_choice = max(rows, key=lambda x: x["IoU"])

        if save_selected_png:
            write_mask_png(seq_out_dir / "selected_by_sam" / f"{key}_sam_best.png", sam_choice["mask"])
            write_mask_png(seq_out_dir / "selected_by_score_B_unweighted" / f"{key}_best.png", b_choice["mask"])
            write_mask_png(seq_out_dir / "selected_by_score_AB_weighted_B_assumption" / f"{key}_best.png", ab_choice["mask"])
            write_mask_png(seq_out_dir / "selected_by_oracle" / f"{key}_oracle_best.png", oracle_choice["mask"])

        per_frame_rows.append({
            "sequence": sequence,
            "frame": key,
            "note": "ok",
            "num_candidates": int(len(rows)),

            "sam_idx": int(sam_choice["idx"]),
            "sam_id": int(sam_choice["id"]),
            "sam_iou": float(sam_choice["IoU"]),
            "sam_score": float(sam_choice["sam_score"]),

            "b_idx": int(b_choice["idx"]),
            "b_id": int(b_choice["id"]),
            "b_unweighted_iou": float(b_choice["IoU"]),
            "b_unweighted_score": float(b_choice["score_B_unweighted"]),

            "ab_idx": int(ab_choice["idx"]),
            "ab_id": int(ab_choice["id"]),
            "ab_iou": float(ab_choice["IoU"]),
            "ab_score": float(ab_choice["score_AB"]),

            "oracle_idx": int(oracle_choice["idx"]),
            "oracle_id": int(oracle_choice["id"]),
            "oracle_iou": float(oracle_choice["IoU"]),

            "ab_delta_vs_sam": float(ab_choice["IoU"] - sam_choice["IoU"]),
            "ab_delta_vs_b": float(ab_choice["IoU"] - b_choice["IoU"]),
            "ab_gap_to_oracle": float(oracle_choice["IoU"] - ab_choice["IoU"]),
            "b_gap_to_oracle": float(oracle_choice["IoU"] - b_choice["IoU"]),

            "ab_same_as_sam": int(ab_choice["idx"] == sam_choice["idx"]),
            "ab_same_as_b": int(ab_choice["idx"] == b_choice["idx"]),
            "ab_same_as_oracle": int(ab_choice["idx"] == oracle_choice["idx"]),
            "b_same_as_oracle": int(b_choice["idx"] == oracle_choice["idx"]),

            "ab_A_match": float(ab_choice["A_match"]),
            "ab_C_match": float(ab_choice["C_match"]),
            "ab_E_match": float(ab_choice["E_match"]),
            "ab_S_match": float(ab_choice["S_match"]),

            "ab_A_share": float(ab_choice["AB_A_share"]),
            "ab_C_share": float(ab_choice["AB_C_share"]),
            "ab_E_share": float(ab_choice["AB_E_share"]),
            "ab_S_share": float(ab_choice["AB_S_share"]),

            "b_A_match": float(b_choice["A_match"]),
            "b_C_match": float(b_choice["C_match"]),
            "b_E_match": float(b_choice["E_match"]),
            "b_S_match": float(b_choice["S_match"]),

            "param_q_border": float(params["q_border"]),
            "param_t_area": float(params["t_area"]),
            "param_c_target": float(params["c_target"]),
            "param_e_target": float(params["e_target"]),
            "param_s_target": float(params["s_target"]),

            "wA": float(weights_vec[0]),
            "wC": float(weights_vec[1]),
            "wE": float(weights_vec[2]),
            "wS": float(weights_vec[3]),
        })

    ok_rows = [r for r in per_frame_rows if r.get("note") == "ok"]

    if not ok_rows:
        return None, per_frame_rows, candidate_rows, "no_valid_frames"

    def vals(col):
        return [float(r[col]) for r in ok_rows if col in r and pd.notna(r[col])]

    sequence_summary = {
        "sequence": sequence,
        "version": "AB",
        "definition": "score_AB = wA*A_match + wC*C_match + wE*E_match + wS*S_match; B optimized assumptions + A default weights",
        "n_frames_evaluated": int(len(ok_rows)),

        "sam_mean_iou": summarize_array(vals("sam_iou"))["mean"],
        "b_unweighted_mean_iou": summarize_array(vals("b_unweighted_iou"))["mean"],
        "ab_mean_iou": summarize_array(vals("ab_iou"))["mean"],
        "oracle_mean_iou": summarize_array(vals("oracle_iou"))["mean"],

        "ab_delta_vs_sam": summarize_array(vals("ab_delta_vs_sam"))["mean"],
        "ab_delta_vs_b": summarize_array(vals("ab_delta_vs_b"))["mean"],
        "ab_gap_to_oracle": summarize_array(vals("ab_gap_to_oracle"))["mean"],
        "b_gap_to_oracle": summarize_array(vals("b_gap_to_oracle"))["mean"],

        "ab_same_as_sam_ratio": summarize_array(vals("ab_same_as_sam"))["mean"],
        "ab_same_as_b_ratio": summarize_array(vals("ab_same_as_b"))["mean"],
        "ab_same_as_oracle_ratio": summarize_array(vals("ab_same_as_oracle"))["mean"],
        "b_same_as_oracle_ratio": summarize_array(vals("b_same_as_oracle"))["mean"],

        "mean_ab_score": summarize_array(vals("ab_score"))["mean"],
        "mean_ab_A_match": summarize_array(vals("ab_A_match"))["mean"],
        "mean_ab_C_match": summarize_array(vals("ab_C_match"))["mean"],
        "mean_ab_E_match": summarize_array(vals("ab_E_match"))["mean"],
        "mean_ab_S_match": summarize_array(vals("ab_S_match"))["mean"],

        "mean_ab_A_share": summarize_array(vals("ab_A_share"))["mean"],
        "mean_ab_C_share": summarize_array(vals("ab_C_share"))["mean"],
        "mean_ab_E_share": summarize_array(vals("ab_E_share"))["mean"],
        "mean_ab_S_share": summarize_array(vals("ab_S_share"))["mean"],

        "param_q_border": float(params["q_border"]),
        "param_t_area": float(params["t_area"]),
        "param_c_target": float(params["c_target"]),
        "param_e_target": float(params["e_target"]),
        "param_s_target": float(params["s_target"]),

        "wA": float(weights_vec[0]),
        "wC": float(weights_vec[1]),
        "wE": float(weights_vec[2]),
        "wS": float(weights_vec[3]),
    }

    return sequence_summary, per_frame_rows, candidate_rows, None


def global_summary_from_sequence_df(seq_df: pd.DataFrame) -> pd.DataFrame:
    ok = seq_df.copy()
    if ok.empty:
        return pd.DataFrame([{
            "num_sequences": 0,
            "num_frames": 0,
        }])

    numeric_cols = [
        "n_frames_evaluated",
        "sam_mean_iou",
        "b_unweighted_mean_iou",
        "ab_mean_iou",
        "oracle_mean_iou",
        "ab_delta_vs_sam",
        "ab_delta_vs_b",
        "ab_gap_to_oracle",
        "b_gap_to_oracle",
        "ab_same_as_sam_ratio",
        "ab_same_as_b_ratio",
        "ab_same_as_oracle_ratio",
        "b_same_as_oracle_ratio",
        "mean_ab_score",
        "mean_ab_A_match",
        "mean_ab_C_match",
        "mean_ab_E_match",
        "mean_ab_S_match",
        "mean_ab_A_share",
        "mean_ab_C_share",
        "mean_ab_E_share",
        "mean_ab_S_share",
    ]
    for col in numeric_cols:
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")

    weights = ok["n_frames_evaluated"].to_numpy(dtype=float)

    def macro(col):
        return float(ok[col].mean()) if col in ok.columns else np.nan

    def weighted(col):
        if col not in ok.columns:
            return np.nan
        vals = ok[col].to_numpy(dtype=float)
        mask = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
        if not mask.any():
            return np.nan
        return float(np.average(vals[mask], weights=weights[mask]))

    row = {
        "num_sequences": int(len(ok)),
        "num_frames": int(ok["n_frames_evaluated"].sum()),

        "macro_sam_mean_iou": macro("sam_mean_iou"),
        "macro_b_unweighted_mean_iou": macro("b_unweighted_mean_iou"),
        "macro_ab_mean_iou": macro("ab_mean_iou"),
        "macro_oracle_mean_iou": macro("oracle_mean_iou"),

        "macro_ab_delta_vs_sam": macro("ab_delta_vs_sam"),
        "macro_ab_delta_vs_b": macro("ab_delta_vs_b"),
        "macro_ab_gap_to_oracle": macro("ab_gap_to_oracle"),

        "weighted_sam_mean_iou": weighted("sam_mean_iou"),
        "weighted_b_unweighted_mean_iou": weighted("b_unweighted_mean_iou"),
        "weighted_ab_mean_iou": weighted("ab_mean_iou"),
        "weighted_oracle_mean_iou": weighted("oracle_mean_iou"),

        "weighted_ab_delta_vs_sam": weighted("ab_delta_vs_sam"),
        "weighted_ab_delta_vs_b": weighted("ab_delta_vs_b"),
        "weighted_ab_gap_to_oracle": weighted("ab_gap_to_oracle"),

        "macro_ab_same_as_b_ratio": macro("ab_same_as_b_ratio"),
        "macro_ab_same_as_oracle_ratio": macro("ab_same_as_oracle_ratio"),
        "weighted_ab_same_as_b_ratio": weighted("ab_same_as_b_ratio"),
        "weighted_ab_same_as_oracle_ratio": weighted("ab_same_as_oracle_ratio"),

        "macro_mean_ab_score": macro("mean_ab_score"),
        "macro_mean_ab_A_match": macro("mean_ab_A_match"),
        "macro_mean_ab_C_match": macro("mean_ab_C_match"),
        "macro_mean_ab_E_match": macro("mean_ab_E_match"),
        "macro_mean_ab_S_match": macro("mean_ab_S_match"),
        "macro_mean_ab_A_share": macro("mean_ab_A_share"),
        "macro_mean_ab_C_share": macro("mean_ab_C_share"),
        "macro_mean_ab_E_share": macro("mean_ab_E_share"),
        "macro_mean_ab_S_share": macro("mean_ab_S_share"),

        "weighted_mean_ab_score": weighted("mean_ab_score"),
        "weighted_mean_ab_A_match": weighted("mean_ab_A_match"),
        "weighted_mean_ab_C_match": weighted("mean_ab_C_match"),
        "weighted_mean_ab_E_match": weighted("mean_ab_E_match"),
        "weighted_mean_ab_S_match": weighted("mean_ab_S_match"),
        "weighted_mean_ab_A_share": weighted("mean_ab_A_share"),
        "weighted_mean_ab_C_share": weighted("mean_ab_C_share"),
        "weighted_mean_ab_E_share": weighted("mean_ab_E_share"),
        "weighted_mean_ab_S_share": weighted("mean_ab_S_share"),
    }

    return pd.DataFrame([row])


def parse_sequence_list(args, data_root: Path) -> List[str]:
    if args.all_sequences:
        return list_available_sequences(data_root)

    if args.sequences:
        return [s.strip() for s in args.sequences.split(",") if s.strip()]

    if args.sequence_file:
        p = Path(args.sequence_file)
        lines = p.read_text(encoding="utf-8").splitlines()
        return [x.strip() for x in lines if x.strip() and not x.strip().startswith("#")]

    return DEFAULT_SEQUENCES


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default=r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results",
        help="Existing SAM/DINO cache root containing sequence/json and sequence/gt_json.",
    )
    parser.add_argument(
        "--b_root",
        type=str,
        default=r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results_B",
        help="B-version output root containing optimized_params_B.json for each sequence.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="",
        help="Output root. Default: <b_root>/_analysis_AB_version",
    )

    parser.add_argument("--wA", type=float, default=DEFAULT_A_WEIGHTS["wA"])
    parser.add_argument("--wC", type=float, default=DEFAULT_A_WEIGHTS["wC"])
    parser.add_argument("--wE", type=float, default=DEFAULT_A_WEIGHTS["wE"])
    parser.add_argument("--wS", type=float, default=DEFAULT_A_WEIGHTS["wS"])

    parser.add_argument(
        "--sequences",
        type=str,
        default="",
        help="Comma-separated sequence names. If omitted, uses the current 33-sequence thesis subset.",
    )
    parser.add_argument(
        "--sequence_file",
        type=str,
        default="",
        help="Optional text file with one sequence name per line.",
    )
    parser.add_argument(
        "--all_sequences",
        action="store_true",
        help="Analyze every sequence folder found in data_root.",
    )

    parser.add_argument(
        "--save_selected_png",
        action="store_true",
        help="Save SAM, B, AB, and oracle selected masks as png files.",
    )
    parser.add_argument(
        "--save_candidate_details",
        action="store_true",
        help="Save per-candidate score table. This can be large.",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    b_root = Path(args.b_root)
    out_root = Path(args.out_root) if args.out_root else b_root / "_analysis_AB_version"

    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")
    if not b_root.exists():
        raise FileNotFoundError(f"b_root does not exist: {b_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    weights_vec = normalize_weights({
        "wA": args.wA,
        "wC": args.wC,
        "wE": args.wE,
        "wS": args.wS,
    })

    sequences = parse_sequence_list(args, data_root)

    sequence_summaries = []
    all_per_frame_rows = []
    all_candidate_rows = []
    skipped_rows = []

    print(f"[INFO] data_root = {data_root}")
    print(f"[INFO] b_root    = {b_root}")
    print(f"[INFO] out_root  = {out_root}")
    print(
        "[INFO] AB weights = "
        f"wA={weights_vec[0]:.4f}, wC={weights_vec[1]:.4f}, "
        f"wE={weights_vec[2]:.4f}, wS={weights_vec[3]:.4f}"
    )
    print(f"[INFO] sequences = {len(sequences)}")

    for i, seq in enumerate(sequences, start=1):
        print(f"[{i}/{len(sequences)}] Evaluating AB: {seq}")

        summary, per_frame_rows, candidate_rows, skip_reason = evaluate_sequence_AB(
            sequence=seq,
            data_root=data_root,
            b_root=b_root,
            out_root=out_root,
            weights_vec=weights_vec,
            save_selected_png=args.save_selected_png,
            save_candidate_details=args.save_candidate_details,
        )

        if summary is None:
            skipped_rows.append({
                "sequence": seq,
                "skip_reason": skip_reason or "unknown",
            })
            continue

        sequence_summaries.append(summary)
        all_per_frame_rows.extend(per_frame_rows)
        all_candidate_rows.extend(candidate_rows)

    seq_df = pd.DataFrame(sequence_summaries)
    frame_df = pd.DataFrame(all_per_frame_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    global_df = global_summary_from_sequence_df(seq_df)

    seq_path = out_root / "AB_per_sequence_summary.csv"
    frame_path = out_root / "AB_per_frame.csv"
    global_path = out_root / "AB_global_summary.csv"
    skipped_path = out_root / "AB_skipped_sequences.csv"

    seq_df.to_csv(seq_path, index=False)
    frame_df.to_csv(frame_path, index=False)
    global_df.to_csv(global_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)

    if args.save_candidate_details:
        cand_df = pd.DataFrame(all_candidate_rows)
        cand_path = out_root / "AB_candidate_details.csv"
        cand_df.to_csv(cand_path, index=False)
        print(f"[DONE] Candidate details saved to: {cand_path}")

    print(f"[DONE] Per-sequence summary saved to: {seq_path}")
    print(f"[DONE] Per-frame results saved to: {frame_path}")
    print(f"[DONE] Global summary saved to: {global_path}")
    print(f"[DONE] Skipped sequences saved to: {skipped_path}")

    print("\n=== AB global summary ===")
    if not global_df.empty:
        print(global_df.to_string(index=False))

    if not seq_df.empty:
        print("\n=== Key per-sequence columns ===")
        key_cols = [
            "sequence",
            "n_frames_evaluated",
            "sam_mean_iou",
            "b_unweighted_mean_iou",
            "ab_mean_iou",
            "oracle_mean_iou",
            "ab_delta_vs_b",
            "ab_gap_to_oracle",
            "ab_same_as_b_ratio",
            "ab_same_as_oracle_ratio",
        ]
        key_cols = [c for c in key_cols if c in seq_df.columns]
        print(seq_df[key_cols].to_string(index=False))


if __name__ == "__main__":
    main()
