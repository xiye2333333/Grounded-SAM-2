"""
Analyze candidate-level B-version term/score correlations with IoU.

This script reads existing SAM/DINO candidate caches and per-sequence optimized B parameters.
It does NOT rerun SAM/DINO. For each frame, it computes every candidate mask's:
  - IoU vs GT
  - A_match, C_match, E_match, S_match
  - score_B = A_match + C_match + E_match + S_match

Then it reports:
  1. Per-frame Spearman rank correlation between each term/score_B and IoU.
  2. Per-frame IoU selected by argmax of each term and score_B.
  3. Per-sequence averages.
  4. Global macro and frame-weighted summary.

Default paths match the user's DAVIS/SAM_results and SAM_results_B layout.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

try:
    import pycocotools.mask as mask_util
except ImportError as exc:
    raise ImportError(
        "pycocotools is required to decode compressed COCO RLE masks. "
        "Install with: pip install pycocotools"
    ) from exc

EPS = 1e-8

# The 33 sequences you selected in the previous step.
DEFAULT_SELECTED_SEQUENCES = [
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

DEFAULT_B_PARAMS = {
    "q_border": 0.25,
    "t_area": 0.25,
    "c_target": 1.0,
    "e_target": 0.0,
    "s_target": 1.0,
}

TERM_COLS = ["A_match", "C_match", "E_match", "S_match"]
SCORE_COLS = ["score_B", "A_match", "C_match", "E_match", "S_match", "sam_score"]


# =========================================================
# Basic file / mask utilities
# =========================================================

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rle_to_mask(rle_obj: Dict[str, Any]) -> np.ndarray:
    """Decode compressed or uncompressed COCO RLE to bool mask."""
    if rle_obj is None:
        raise ValueError("RLE object is None")
    size = rle_obj.get("size")
    counts = rle_obj.get("counts")
    if size is None or counts is None:
        raise ValueError("Invalid RLE: missing size/counts")

    h, w = int(size[0]), int(size[1])

    if isinstance(counts, str):
        rle = {"size": [h, w], "counts": counts.encode("utf-8")}
        m = mask_util.decode(rle)
        if m.ndim == 3:
            m = m[:, :, 0]
        return m.astype(bool)

    if isinstance(counts, bytes):
        rle = {"size": [h, w], "counts": counts}
        m = mask_util.decode(rle)
        if m.ndim == 3:
            m = m[:, :, 0]
        return m.astype(bool)

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
        # COCO uncompressed RLE is column-major.
        return arr.reshape((w, h)).T.astype(bool)

    raise TypeError(f"Unsupported RLE counts type: {type(counts)}")


def load_gt_mask_from_gt_json(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    data = read_json(path)
    rle = data.get("segmentation_rle")
    if not isinstance(rle, dict):
        return None
    return rle_to_mask(rle)


def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def frame_key_from_result_json(path: Path) -> str:
    return path.stem.replace("_result", "")


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


# =========================================================
# B-version feature / score functions
# =========================================================

def compute_distance_transform(h: int, w: int) -> np.ndarray:
    border_mask = np.zeros((h, w), np.uint8)
    border_mask[1:-1, 1:-1] = 1
    return cv2.distanceTransform(border_mask, distanceType=cv2.DIST_L2, maskSize=5)


def area_term_parabola(x: float, d: float, eps: float = 1e-6) -> float:
    d = float(np.clip(d, eps, 1.0))
    x = float(np.clip(x, 0.0, 1.0))
    val = 1.0 - ((x - d) / d) ** 2
    return float(np.clip(val, 0.0, 1.0))


def compute_silhouette_score_v2(mask: np.ndarray) -> float:
    mask_u8 = (mask.astype(np.uint8) > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                fragmentation = 1.0 - largest_area / float(np.sum(large_areas))

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


def mask_features(mask_bool: np.ndarray, W: int, H: int, dist_map: np.ndarray,
                  d_max: float, q_border: float, t_area: float) -> np.ndarray:
    img_area = W * H
    cx, cy = W / 2, H / 2

    area_px = int(mask_bool.sum())
    if area_px <= 0:
        return np.asarray([0.0, 0.0, 0.0, 0.0], dtype=float)

    # A: area prior / area target compatibility.
    A_raw = area_px / img_area
    A = area_term_parabola(A_raw, t_area)

    # C: centeredness observation. 1=centered, 0=far from center.
    ys, xs = np.where(mask_bool)
    mx, my = xs.mean(), ys.mean()
    Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
    C = 1.0 - float(np.clip(Dp, 0.0, 1.0))

    # E_dist: border-distance support. 1=far from image border, 0=near border.
    q = float(np.quantile(dist_map[mask_bool], q_border))
    E_dist = float(np.clip(q / d_max, 0.0, 1.0))

    # S: silhouette quality observation.
    Sil = compute_silhouette_score_v2(mask_bool)

    return np.asarray([A, C, E_dist, Sil], dtype=float)


def mask_features_b(mask_bool: np.ndarray, W: int, H: int, dist_map: np.ndarray,
                    d_max: float, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      observed: [A_score, C_obs, edge_proximity, Sil_obs]
      matches:  [A_match, C_match, E_match, S_match]

    Note: B target e_target is edge proximity. The original E observation is
    border-distance support, so edge_proximity = 1 - E_dist.
    """
    q_border = float(np.clip(params.get("q_border", DEFAULT_B_PARAMS["q_border"]), 0.0, 1.0))
    t_area = float(np.clip(params.get("t_area", DEFAULT_B_PARAMS["t_area"]), 1e-6, 1.0))
    c_target = float(np.clip(params.get("c_target", DEFAULT_B_PARAMS["c_target"]), 0.0, 1.0))
    e_target = float(np.clip(params.get("e_target", DEFAULT_B_PARAMS["e_target"]), 0.0, 1.0))
    s_target = float(np.clip(params.get("s_target", DEFAULT_B_PARAMS["s_target"]), 0.0, 1.0))

    old = mask_features(mask_bool, W, H, dist_map, d_max, q_border, t_area)
    A_score = float(old[0])
    C_obs = float(old[1])
    E_dist = float(old[2])
    edge_proximity = 1.0 - E_dist
    Sil_obs = float(old[3])

    A_match = A_score
    C_match = 1.0 - abs(c_target - C_obs)
    E_match = 1.0 - abs(e_target - edge_proximity)
    S_match = 1.0 - abs(s_target - Sil_obs)

    observed = np.asarray([A_score, C_obs, edge_proximity, Sil_obs], dtype=float)
    matches = np.asarray([
        np.clip(A_match, 0.0, 1.0),
        np.clip(C_match, 0.0, 1.0),
        np.clip(E_match, 0.0, 1.0),
        np.clip(S_match, 0.0, 1.0),
    ], dtype=float)
    return observed, matches


# =========================================================
# Statistics helpers
# =========================================================

def spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. Returns NaN when undefined."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2 or y.size < 2:
        return np.nan
    if np.nanstd(x) <= EPS or np.nanstd(y) <= EPS:
        return np.nan

    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)

    if np.nanstd(rx) <= EPS or np.nanstd(ry) <= EPS:
        return np.nan

    return float(np.corrcoef(rx, ry)[0, 1])


def argmax_first(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return -1
    return int(np.nanargmax(values))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


# =========================================================
# Sequence loading / analysis
# =========================================================

def load_optimized_params(b_seq_dir: Path) -> Optional[Dict[str, float]]:
    opt_path = b_seq_dir / "optimized_params_B.json"
    if not opt_path.exists():
        return None
    obj = read_json(opt_path)
    if obj.get("skipped", False):
        return None
    optimized = obj.get("optimized", {})
    if not optimized:
        return None
    return {
        "q_border": safe_float(optimized.get("q_border", DEFAULT_B_PARAMS["q_border"])),
        "t_area": safe_float(optimized.get("t_area", DEFAULT_B_PARAMS["t_area"])),
        "c_target": safe_float(optimized.get("c_target", DEFAULT_B_PARAMS["c_target"])),
        "e_target": safe_float(optimized.get("e_target", DEFAULT_B_PARAMS["e_target"])),
        "s_target": safe_float(optimized.get("s_target", DEFAULT_B_PARAMS["s_target"])),
    }


def list_available_sequences(data_root: Path, b_root: Path, mode: str) -> List[str]:
    if mode == "selected":
        candidates = DEFAULT_SELECTED_SEQUENCES
    elif mode == "all_optimized":
        candidates = sorted([p.parent.name for p in b_root.rglob("optimized_params_B.json")])
    elif mode == "all_data":
        candidates = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    else:
        raise ValueError(f"Unknown sequence mode: {mode}")

    seqs = []
    for seq in candidates:
        if not (data_root / seq / "json").exists():
            continue
        if load_optimized_params(b_root / seq) is None:
            continue
        seqs.append(seq)
    return seqs


def analyze_frame(seq: str, result_path: Path, data_seq_dir: Path,
                  params: Dict[str, float], save_candidates: bool) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    key = frame_key_from_result_json(result_path)
    gt_path = data_seq_dir / "gt_json" / f"{key}_gt.json"
    gt = load_gt_mask_from_gt_json(gt_path)
    if gt is None:
        return None, []

    try:
        data = read_json(result_path)
    except Exception as e:
        return {
            "sequence": seq,
            "frame": key,
            "status": "bad_result_json",
            "note": str(e),
        }, []

    anns = data.get("annotations", [])
    if not anns:
        return {
            "sequence": seq,
            "frame": key,
            "status": "no_candidates",
            "note": "annotations is empty",
        }, []

    H = int(data.get("img_height", gt.shape[0]))
    W = int(data.get("img_width", gt.shape[1]))
    if gt.shape[:2] != (H, W):
        H, W = gt.shape[:2]

    dist_map = compute_distance_transform(H, W)
    d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

    raw_scores = data.get("raw_scores", None)
    rows = []

    for idx, ann in enumerate(anns):
        try:
            mask = rle_to_mask(ann["segmentation"])
        except Exception:
            continue
        if mask.shape[:2] != gt.shape[:2]:
            continue

        observed, matches = mask_features_b(mask, W, H, dist_map, d_max, params)
        score_B = float(matches.sum())
        iou = compute_iou(mask, gt)

        if raw_scores is not None and idx < len(raw_scores):
            sam_score = safe_float(raw_scores[idx])
        else:
            sam_score = safe_float(ann.get("sam_score", np.nan))

        rows.append({
            "sequence": seq,
            "frame": key,
            "candidate_order": idx,
            "candidate_id": int(ann.get("id", idx)),
            "box_id": int(ann.get("box_id", -1)),
            "rank_in_box": int(ann.get("rank_in_box", -1)),
            "sam_score": sam_score,
            "IoU": float(iou),
            "score_B": score_B,
            "A_match": float(matches[0]),
            "C_match": float(matches[1]),
            "E_match": float(matches[2]),
            "S_match": float(matches[3]),
            "A_obs_or_score": float(observed[0]),
            "C_obs": float(observed[1]),
            "edge_proximity_obs": float(observed[2]),
            "S_obs": float(observed[3]),
            "param_q_border": float(params["q_border"]),
            "param_t_area": float(params["t_area"]),
            "param_c_target": float(params["c_target"]),
            "param_e_target": float(params["e_target"]),
            "param_s_target": float(params["s_target"]),
        })

    if not rows:
        return {
            "sequence": seq,
            "frame": key,
            "status": "no_valid_candidates",
            "note": "all candidates failed to decode or shape mismatch",
        }, []

    df = pd.DataFrame(rows)
    n = len(df)

    # Choices.
    idx_oracle = argmax_first(df["IoU"].to_numpy())
    idx_sam = argmax_first(df["sam_score"].to_numpy())
    idx_B = argmax_first(df["score_B"].to_numpy())
    idx_A = argmax_first(df["A_match"].to_numpy())
    idx_C = argmax_first(df["C_match"].to_numpy())
    idx_E = argmax_first(df["E_match"].to_numpy())
    idx_S = argmax_first(df["S_match"].to_numpy())

    def choice_record(prefix: str, idx: int) -> Dict[str, Any]:
        if idx < 0:
            return {f"{prefix}_candidate_id": np.nan, f"{prefix}_iou": np.nan}
        r = df.iloc[idx]
        return {
            f"{prefix}_candidate_id": int(r["candidate_id"]),
            f"{prefix}_candidate_order": int(r["candidate_order"]),
            f"{prefix}_iou": float(r["IoU"]),
            f"{prefix}_score_B": float(r["score_B"]),
            f"{prefix}_A_match": float(r["A_match"]),
            f"{prefix}_C_match": float(r["C_match"]),
            f"{prefix}_E_match": float(r["E_match"]),
            f"{prefix}_S_match": float(r["S_match"]),
            f"{prefix}_sam_score": float(r["sam_score"]),
        }

    # Correlations within this frame.
    frame = {
        "sequence": seq,
        "frame": key,
        "status": "ok",
        "n_candidates": int(n),
        "param_q_border": float(params["q_border"]),
        "param_t_area": float(params["t_area"]),
        "param_c_target": float(params["c_target"]),
        "param_e_target": float(params["e_target"]),
        "param_s_target": float(params["s_target"]),
    }

    for col in SCORE_COLS:
        if col in df.columns:
            frame[f"spearman_{col}_vs_IoU"] = spearman_safe(df[col].to_numpy(), df["IoU"].to_numpy())
        else:
            frame[f"spearman_{col}_vs_IoU"] = np.nan

    # IoU if selected by each score/term.
    frame.update(choice_record("oracle", idx_oracle))
    frame.update(choice_record("sam", idx_sam))
    frame.update(choice_record("B", idx_B))
    frame.update(choice_record("A_only", idx_A))
    frame.update(choice_record("C_only", idx_C))
    frame.update(choice_record("E_only", idx_E))
    frame.update(choice_record("S_only", idx_S))

    # Gaps.
    frame["B_gap_to_oracle"] = frame["oracle_iou"] - frame["B_iou"]
    frame["sam_gap_to_oracle"] = frame["oracle_iou"] - frame["sam_iou"]
    frame["B_delta_vs_sam"] = frame["B_iou"] - frame["sam_iou"]
    frame["B_equals_oracle"] = int(frame["B_candidate_id"] == frame["oracle_candidate_id"])
    frame["sam_equals_oracle"] = int(frame["sam_candidate_id"] == frame["oracle_candidate_id"])
    frame["B_equals_sam"] = int(frame["B_candidate_id"] == frame["sam_candidate_id"])

    # Selected B term share / dominance diagnostics.
    b_row = df.iloc[idx_B]
    b_score = float(b_row["score_B"])
    if b_score > EPS:
        shares = {
            "A": float(b_row["A_match"] / b_score),
            "C": float(b_row["C_match"] / b_score),
            "E": float(b_row["E_match"] / b_score),
            "S": float(b_row["S_match"] / b_score),
        }
    else:
        shares = {"A": np.nan, "C": np.nan, "E": np.nan, "S": np.nan}

    for k, v in shares.items():
        frame[f"B_selected_{k}_share"] = v
    valid_shares = {k: v for k, v in shares.items() if np.isfinite(v)}
    if valid_shares:
        dominant_term = max(valid_shares, key=valid_shares.get)
        frame["B_selected_dominant_term"] = dominant_term
        frame["B_selected_dominant_share"] = valid_shares[dominant_term]
    else:
        frame["B_selected_dominant_term"] = ""
        frame["B_selected_dominant_share"] = np.nan

    # Useful quick comparison: did unweighted score_B underperform a single term?
    term_iou_values = {
        "A_only": frame["A_only_iou"],
        "C_only": frame["C_only_iou"],
        "E_only": frame["E_only_iou"],
        "S_only": frame["S_only_iou"],
    }
    best_single_term = max(term_iou_values, key=lambda k: term_iou_values[k] if np.isfinite(term_iou_values[k]) else -1)
    frame["best_single_term_by_iou"] = best_single_term
    frame["best_single_term_iou"] = float(term_iou_values[best_single_term])
    frame["B_gap_to_best_single_term"] = frame["best_single_term_iou"] - frame["B_iou"]

    return frame, (rows if save_candidates else [])


def summarize_sequence(seq: str, frame_df: pd.DataFrame) -> Dict[str, Any]:
    ok = frame_df[frame_df["status"].eq("ok")].copy()
    row: Dict[str, Any] = {
        "sequence": seq,
        "n_frames_ok": int(len(ok)),
        "n_candidates_total": int(ok["n_candidates"].sum()) if not ok.empty else 0,
        "mean_candidates_per_frame": float(ok["n_candidates"].mean()) if not ok.empty else np.nan,
    }
    if ok.empty:
        return row

    # Mean IoU for each selection rule.
    for prefix in ["sam", "B", "oracle", "A_only", "C_only", "E_only", "S_only", "best_single_term"]:
        col = f"{prefix}_iou" if prefix != "best_single_term" else "best_single_term_iou"
        if col in ok.columns:
            row[f"mean_{col}"] = float(pd.to_numeric(ok[col], errors="coerce").mean())

    row["mean_B_delta_vs_sam"] = float(ok["B_delta_vs_sam"].mean())
    row["mean_B_gap_to_oracle"] = float(ok["B_gap_to_oracle"].mean())
    row["mean_sam_gap_to_oracle"] = float(ok["sam_gap_to_oracle"].mean())
    row["mean_B_gap_to_best_single_term"] = float(ok["B_gap_to_best_single_term"].mean())

    row["B_equals_oracle_ratio"] = float(ok["B_equals_oracle"].mean())
    row["sam_equals_oracle_ratio"] = float(ok["sam_equals_oracle"].mean())
    row["B_equals_sam_ratio"] = float(ok["B_equals_sam"].mean())

    # Spearman averages.
    for col in [c for c in ok.columns if c.startswith("spearman_")]:
        row[f"mean_{col}"] = float(pd.to_numeric(ok[col], errors="coerce").mean())
        row[f"median_{col}"] = float(pd.to_numeric(ok[col], errors="coerce").median())
        row[f"valid_n_{col}"] = int(pd.to_numeric(ok[col], errors="coerce").notna().sum())

    # Selected B term scores and shares.
    for term in ["A", "C", "E", "S"]:
        match_col = f"B_{term}_match"
        share_col = f"B_selected_{term}_share"
        if match_col in ok.columns:
            row[f"mean_B_selected_{term}_match"] = float(ok[match_col].mean())
        if share_col in ok.columns:
            row[f"mean_B_selected_{term}_share"] = float(ok[share_col].mean())

    if "B_selected_dominant_share" in ok.columns:
        row["mean_B_selected_dominant_share"] = float(ok["B_selected_dominant_share"].mean())
    if "B_selected_dominant_term" in ok.columns:
        counts = ok["B_selected_dominant_term"].value_counts(normalize=True)
        for term in ["A", "C", "E", "S"]:
            row[f"B_selected_dominant_{term}_ratio"] = float(counts.get(term, 0.0))

    if "best_single_term_by_iou" in ok.columns:
        counts = ok["best_single_term_by_iou"].value_counts(normalize=True)
        for term_name in ["A_only", "C_only", "E_only", "S_only"]:
            row[f"best_single_term_{term_name}_ratio"] = float(counts.get(term_name, 0.0))

    # Parameters copied from first frame.
    for p in ["param_q_border", "param_t_area", "param_c_target", "param_e_target", "param_s_target"]:
        if p in ok.columns:
            row[p] = float(ok[p].iloc[0])

    return row


def make_global_summary(seq_summary: pd.DataFrame) -> pd.DataFrame:
    if seq_summary.empty:
        return pd.DataFrame()
    ok = seq_summary[seq_summary["n_frames_ok"] > 0].copy()
    if ok.empty:
        return pd.DataFrame()

    weights = ok["n_frames_ok"].to_numpy(dtype=float)
    metrics = [
        "mean_sam_iou",
        "mean_B_iou",
        "mean_oracle_iou",
        "mean_A_only_iou",
        "mean_C_only_iou",
        "mean_E_only_iou",
        "mean_S_only_iou",
        "mean_best_single_term_iou",
        "mean_B_delta_vs_sam",
        "mean_B_gap_to_oracle",
        "mean_B_gap_to_best_single_term",
        "B_equals_oracle_ratio",
        "sam_equals_oracle_ratio",
        "B_equals_sam_ratio",
        "mean_spearman_score_B_vs_IoU",
        "mean_spearman_A_match_vs_IoU",
        "mean_spearman_C_match_vs_IoU",
        "mean_spearman_E_match_vs_IoU",
        "mean_spearman_S_match_vs_IoU",
        "mean_B_selected_A_share",
        "mean_B_selected_C_share",
        "mean_B_selected_E_share",
        "mean_B_selected_S_share",
        "mean_B_selected_dominant_share",
    ]

    out = {
        "n_sequences": int(len(ok)),
        "n_frames_total": int(ok["n_frames_ok"].sum()),
        "n_candidates_total": int(ok["n_candidates_total"].sum()),
    }

    for m in metrics:
        if m in ok.columns:
            vals = pd.to_numeric(ok[m], errors="coerce").to_numpy(dtype=float)
            out[f"macro_{m}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            out[f"frame_weighted_{m}"] = weighted_mean(vals, weights)

    return pd.DataFrame([out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str,
                    default=r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results",
                    help="Existing SAM/DINO cache root containing per-sequence json/gt_json.")
    ap.add_argument("--b_root", type=str,
                    default=r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results_B",
                    help="B-version output root containing optimized_params_B.json.")
    ap.add_argument("--output_dir", type=str, default="",
                    help="Default: <b_root>/_analysis_correlations")
    ap.add_argument("--sequence_mode", type=str, default="selected",
                    choices=["selected", "all_optimized", "all_data"],
                    help="selected = the 33 sequences listed in this script; all_optimized = all optimized B dirs.")
    ap.add_argument("--only_seq", type=str, default="",
                    help="Optional comma-separated sequence names, e.g. bear,breakdance")
    ap.add_argument("--save_candidate_details", action="store_true",
                    help="Also save candidate-level table. This can be large but useful for debugging.")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    b_root = Path(args.b_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    if not b_root.exists():
        raise FileNotFoundError(f"b_root not found: {b_root}")

    output_dir = Path(args.output_dir) if args.output_dir else b_root / "_analysis_correlations"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.only_seq.strip():
        seqs = [s.strip() for s in args.only_seq.split(",") if s.strip()]
        seqs = [s for s in seqs if (data_root / s / "json").exists() and load_optimized_params(b_root / s) is not None]
    else:
        seqs = list_available_sequences(data_root, b_root, args.sequence_mode)

    print(f"[INFO] Number of sequences to analyze: {len(seqs)}")
    if not seqs:
        print("[WARN] No valid sequences found.")
        return

    all_frame_rows = []
    all_candidate_rows = []
    seq_summary_rows = []
    skipped_sequences = []

    for seq in seqs:
        data_seq_dir = data_root / seq
        b_seq_dir = b_root / seq
        params = load_optimized_params(b_seq_dir)
        if params is None:
            skipped_sequences.append({"sequence": seq, "reason": "missing optimized params"})
            continue

        result_files = sorted((data_seq_dir / "json").glob("*_result.json"))
        if not result_files:
            skipped_sequences.append({"sequence": seq, "reason": "no result json files"})
            continue

        print(f"[SEQ] {seq}: {len(result_files)} result files")
        seq_frame_rows = []

        for result_path in result_files:
            frame_row, candidate_rows = analyze_frame(
                seq=seq,
                result_path=result_path,
                data_seq_dir=data_seq_dir,
                params=params,
                save_candidates=args.save_candidate_details,
            )
            if frame_row is not None:
                seq_frame_rows.append(frame_row)
                all_frame_rows.append(frame_row)
            if candidate_rows:
                all_candidate_rows.extend(candidate_rows)

        if seq_frame_rows:
            seq_frame_df = pd.DataFrame(seq_frame_rows)
            seq_summary_rows.append(summarize_sequence(seq, seq_frame_df))
        else:
            skipped_sequences.append({"sequence": seq, "reason": "no usable frames"})

    frame_df = pd.DataFrame(all_frame_rows)
    seq_summary_df = pd.DataFrame(seq_summary_rows)
    global_df = make_global_summary(seq_summary_df)
    skipped_df = pd.DataFrame(skipped_sequences)

    frame_path = output_dir / "B_candidate_correlation_per_frame.csv"
    seq_path = output_dir / "B_candidate_correlation_per_sequence.csv"
    global_path = output_dir / "B_candidate_correlation_global_summary.csv"
    skipped_path = output_dir / "B_candidate_correlation_skipped_sequences.csv"

    frame_df.to_csv(frame_path, index=False)
    seq_summary_df.to_csv(seq_path, index=False)
    global_df.to_csv(global_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)

    print(f"[DONE] Per-frame correlation table saved to: {frame_path}")
    print(f"[DONE] Per-sequence summary saved to: {seq_path}")
    print(f"[DONE] Global summary saved to: {global_path}")
    print(f"[DONE] Skipped sequence list saved to: {skipped_path}")

    if args.save_candidate_details:
        candidate_df = pd.DataFrame(all_candidate_rows)
        cand_path = output_dir / "B_candidate_correlation_candidate_details.csv"
        candidate_df.to_csv(cand_path, index=False)
        print(f"[DONE] Candidate-level details saved to: {cand_path}")

    print("\n=== Global summary ===")
    if not global_df.empty:
        print(global_df.to_string(index=False))

    print("\n=== Per-sequence key columns ===")
    if not seq_summary_df.empty:
        key_cols = [
            "sequence",
            "n_frames_ok",
            "mean_sam_iou",
            "mean_B_iou",
            "mean_oracle_iou",
            "mean_B_gap_to_oracle",
            "mean_B_gap_to_best_single_term",
            "mean_spearman_score_B_vs_IoU",
            "mean_spearman_A_match_vs_IoU",
            "mean_spearman_C_match_vs_IoU",
            "mean_spearman_E_match_vs_IoU",
            "mean_spearman_S_match_vs_IoU",
            "mean_B_selected_A_share",
            "mean_B_selected_C_share",
            "mean_B_selected_E_share",
            "mean_B_selected_S_share",
            "B_selected_dominant_A_ratio",
            "B_selected_dominant_C_ratio",
            "B_selected_dominant_E_ratio",
            "B_selected_dominant_S_ratio",
        ]
        key_cols = [c for c in key_cols if c in seq_summary_df.columns]
        print(seq_summary_df[key_cols].to_string(index=False))


if __name__ == "__main__":
    main()
