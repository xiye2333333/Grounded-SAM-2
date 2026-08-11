#!/usr/bin/env python3
"""
Analyze A-version candidate ranking behavior on DAVIS/SAM_results cache.

This script mirrors the B-version candidate-correlation analysis, but uses
A-version terms and weighted A score:

    score_A = wA*A + wC*C + wE*E + wSil*Sil

For each sequence and frame, it computes all candidate masks' IoU, A/C/E/Sil
term values, score_A, and raw SAM score. It then reports:

1. Per-frame Spearman rank correlation between each term/score and IoU.
2. Per-frame selected IoU from argmax(A), argmax(C), argmax(E), argmax(Sil),
   argmax(score_A), and argmax(SAM raw score).
3. Per-sequence means and global macro/frame-weighted means.
4. Contribution shares of terms in A-selected masks.

Default sequence list matches the 33 sequences used in the B analysis.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None

EPS = 1e-8

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

DEFAULT_A_PARAMS = {
    "wA": 0.15,
    "wC": 0.20,
    "wE": 0.40,
    "wSil": 0.25,
    "t_area": 0.20,
    "q_border": 0.20,
}

TERM_COLS = ["A", "C", "E", "Sil"]
SCORE_COLS = ["score_A", "A", "C", "E", "Sil", "sam_score"]


def decode_rle(segmentation_rle: Dict[str, Any]) -> np.ndarray:
    """Decode COCO-style RLE from JSON into a boolean mask."""
    if segmentation_rle is None:
        raise ValueError("segmentation_rle is None")

    size = segmentation_rle.get("size", None)
    counts = segmentation_rle.get("counts", None)
    if size is None or counts is None:
        raise ValueError("Invalid RLE: missing size or counts")

    h, w = int(size[0]), int(size[1])

    if isinstance(counts, str):
        if mask_utils is None:
            raise ImportError("pycocotools is required for compressed RLE counts")
        rle = {"size": [h, w], "counts": counts.encode("utf-8")}
        return mask_utils.decode(rle).astype(bool)

    if isinstance(counts, bytes):
        if mask_utils is None:
            raise ImportError("pycocotools is required for compressed RLE counts")
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
        return arr.reshape((w, h)).T.astype(bool)

    raise TypeError(f"Unsupported RLE counts type: {type(counts)}")


def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


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


def mask_features_a(mask_bool: np.ndarray, W: int, H: int, dist_map: np.ndarray,
                    d_max: float, q_border: float, t_area: float) -> np.ndarray:
    img_area = W * H
    cx, cy = W / 2, H / 2

    area_px = int(mask_bool.sum())
    if area_px <= 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    # A: area compatibility with expected area ratio.
    A_raw = area_px / img_area
    A = area_term_parabola(A_raw, t_area)

    # C: centeredness.
    ys, xs = np.where(mask_bool)
    mx, my = xs.mean(), ys.mean()
    Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
    C = 1.0 - float(np.clip(Dp, 0.0, 1.0))

    # E: interior support / distance from image border.
    if np.count_nonzero(mask_bool) == 0:
        E = 0.0
    else:
        q = float(np.quantile(dist_map[mask_bool], q_border))
        E = float(np.clip(q / d_max, 0.0, 1.0))

    # Sil: shape quality.
    Sil = compute_silhouette_score_v2(mask_bool)
    return np.array([A, C, E, Sil], dtype=float)


def normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= EPS:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    return w / s


def score_a(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return features @ weights


def load_gt_from_json(gt_path: Path) -> Optional[np.ndarray]:
    try:
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        if "segmentation_rle" not in data:
            return None
        return decode_rle(data["segmentation_rle"])
    except Exception:
        return None


def load_candidates_from_result_json(result_path: Path) -> Tuple[List[Dict[str, Any]], Optional[int], Optional[int]]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    anns = data.get("annotations", [])
    H = data.get("img_height", None)
    W = data.get("img_width", None)
    candidates = []
    for idx, ann in enumerate(anns):
        try:
            mask = decode_rle(ann["segmentation"])
        except Exception:
            continue
        candidates.append({
            "candidate_id": int(ann.get("id", idx)),
            "box_id": int(ann.get("box_id", -1)),
            "rank_in_box": int(ann.get("rank_in_box", -1)),
            "sam_score": float(ann.get("sam_score", np.nan)),
            "mask": mask,
        })
    return candidates, H, W


def spearman_corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.nan
    if np.nanstd(x) <= EPS or np.nanstd(y) <= EPS:
        return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.nanstd(rx) <= EPS or np.nanstd(ry) <= EPS:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze_frame(seq: str, frame_key: str, result_path: Path, gt_path: Path,
                  weights: np.ndarray, q_border: float, t_area: float,
                  save_candidate_details: bool = False):
    gt = load_gt_from_json(gt_path)
    if gt is None:
        return None, [], "missing_or_bad_gt"

    try:
        candidates, H_json, W_json = load_candidates_from_result_json(result_path)
    except Exception as e:
        return None, [], f"bad_result_json: {e}"

    if not candidates:
        return None, [], "no_candidates"

    H, W = gt.shape
    dist_map = compute_distance_transform(H, W)
    d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

    rows = []
    for c in candidates:
        mask = c["mask"]
        if mask.shape != gt.shape:
            continue
        feats = mask_features_a(mask, W, H, dist_map, d_max, q_border, t_area)
        sA = float(score_a(feats[None, :], weights)[0])
        iou = compute_iou(mask, gt)
        weighted_terms = feats * weights
        rows.append({
            "sequence": seq,
            "frame": frame_key,
            "candidate_id": c["candidate_id"],
            "box_id": c["box_id"],
            "rank_in_box": c["rank_in_box"],
            "sam_score": c["sam_score"],
            "IoU": iou,
            "score_A": sA,
            "A": float(feats[0]),
            "C": float(feats[1]),
            "E": float(feats[2]),
            "Sil": float(feats[3]),
            "wA_A": float(weighted_terms[0]),
            "wC_C": float(weighted_terms[1]),
            "wE_E": float(weighted_terms[2]),
            "wSil_Sil": float(weighted_terms[3]),
        })

    if not rows:
        return None, [], "no_valid_candidates"

    df = pd.DataFrame(rows)

    def idxmax_col(col):
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() == 0:
            return None
        return int(vals.idxmax())

    idx_oracle = idxmax_col("IoU")
    idx_A = idxmax_col("score_A")
    idx_sam = idxmax_col("sam_score")
    idx_A_only = idxmax_col("A")
    idx_C_only = idxmax_col("C")
    idx_E_only = idxmax_col("E")
    idx_Sil_only = idxmax_col("Sil")

    selected = {
        "oracle": idx_oracle,
        "A_score": idx_A,
        "SAM": idx_sam,
        "A_only": idx_A_only,
        "C_only": idx_C_only,
        "E_only": idx_E_only,
        "Sil_only": idx_Sil_only,
    }

    def get_iou(idx):
        return float(df.loc[idx, "IoU"]) if idx is not None else np.nan

    def get_score(idx, col):
        return float(df.loc[idx, col]) if idx is not None and col in df.columns else np.nan

    # Term shares for A-score selected mask. Use weighted contribution share.
    if idx_A is not None:
        score_val = max(float(df.loc[idx_A, "score_A"]), EPS)
        A_share = float(df.loc[idx_A, "wA_A"] / score_val)
        C_share = float(df.loc[idx_A, "wC_C"] / score_val)
        E_share = float(df.loc[idx_A, "wE_E"] / score_val)
        Sil_share = float(df.loc[idx_A, "wSil_Sil"] / score_val)
        shares = np.asarray([A_share, C_share, E_share, Sil_share], dtype=float)
        dominant_idx = int(np.nanargmax(shares))
        dominant_term = ["A", "C", "E", "Sil"][dominant_idx]
        dominant_share = float(shares[dominant_idx])
    else:
        A_share = C_share = E_share = Sil_share = dominant_share = np.nan
        dominant_term = ""

    frame_row = {
        "sequence": seq,
        "frame": frame_key,
        "num_candidates": int(len(df)),
        "rho_score_A_vs_IoU": spearman_corr(df["score_A"], df["IoU"]),
        "rho_A_vs_IoU": spearman_corr(df["A"], df["IoU"]),
        "rho_C_vs_IoU": spearman_corr(df["C"], df["IoU"]),
        "rho_E_vs_IoU": spearman_corr(df["E"], df["IoU"]),
        "rho_Sil_vs_IoU": spearman_corr(df["Sil"], df["IoU"]),
        "rho_sam_score_vs_IoU": spearman_corr(df["sam_score"], df["IoU"]),
        "oracle_iou": get_iou(idx_oracle),
        "A_score_iou": get_iou(idx_A),
        "sam_iou": get_iou(idx_sam),
        "A_only_iou": get_iou(idx_A_only),
        "C_only_iou": get_iou(idx_C_only),
        "E_only_iou": get_iou(idx_E_only),
        "Sil_only_iou": get_iou(idx_Sil_only),
        "A_gap_to_oracle": get_iou(idx_oracle) - get_iou(idx_A),
        "A_delta_vs_sam": get_iou(idx_A) - get_iou(idx_sam),
        "A_gap_to_best_single_term": max(
            get_iou(idx_A_only), get_iou(idx_C_only), get_iou(idx_E_only), get_iou(idx_Sil_only)
        ) - get_iou(idx_A),
        "A_selected_candidate_id": int(df.loc[idx_A, "candidate_id"]) if idx_A is not None else -1,
        "sam_selected_candidate_id": int(df.loc[idx_sam, "candidate_id"]) if idx_sam is not None else -1,
        "oracle_candidate_id": int(df.loc[idx_oracle, "candidate_id"]) if idx_oracle is not None else -1,
        "A_selected_score_A": get_score(idx_A, "score_A"),
        "A_selected_A": get_score(idx_A, "A"),
        "A_selected_C": get_score(idx_A, "C"),
        "A_selected_E": get_score(idx_A, "E"),
        "A_selected_Sil": get_score(idx_A, "Sil"),
        "A_selected_wA_A": get_score(idx_A, "wA_A"),
        "A_selected_wC_C": get_score(idx_A, "wC_C"),
        "A_selected_wE_E": get_score(idx_A, "wE_E"),
        "A_selected_wSil_Sil": get_score(idx_A, "wSil_Sil"),
        "A_selected_A_share": A_share,
        "A_selected_C_share": C_share,
        "A_selected_E_share": E_share,
        "A_selected_Sil_share": Sil_share,
        "A_selected_dominant_term": dominant_term,
        "A_selected_dominant_share": dominant_share,
    }

    cand_rows = df.to_dict(orient="records") if save_candidate_details else []
    return frame_row, cand_rows, "ok"


def summarize_sequence(seq: str, frame_rows: List[Dict[str, Any]], params: Dict[str, float]) -> Dict[str, Any]:
    df = pd.DataFrame(frame_rows)
    out = {
        "sequence": seq,
        "n_frames": int(len(df)),
        "wA": params["wA"],
        "wC": params["wC"],
        "wE": params["wE"],
        "wSil": params["wSil"],
        "t_area": params["t_area"],
        "q_border": params["q_border"],
    }
    if df.empty:
        return out

    mean_cols = [
        "rho_score_A_vs_IoU", "rho_A_vs_IoU", "rho_C_vs_IoU", "rho_E_vs_IoU",
        "rho_Sil_vs_IoU", "rho_sam_score_vs_IoU", "oracle_iou", "A_score_iou",
        "sam_iou", "A_only_iou", "C_only_iou", "E_only_iou", "Sil_only_iou",
        "A_gap_to_oracle", "A_delta_vs_sam", "A_gap_to_best_single_term",
        "A_selected_score_A", "A_selected_A", "A_selected_C", "A_selected_E", "A_selected_Sil",
        "A_selected_wA_A", "A_selected_wC_C", "A_selected_wE_E", "A_selected_wSil_Sil",
        "A_selected_A_share", "A_selected_C_share", "A_selected_E_share", "A_selected_Sil_share",
        "A_selected_dominant_share",
    ]
    for col in mean_cols:
        out[f"mean_{col}"] = pd.to_numeric(df[col], errors="coerce").mean() if col in df.columns else np.nan

    # How often each term is the largest weighted contribution in A-selected mask.
    if "A_selected_dominant_term" in df.columns:
        for term in ["A", "C", "E", "Sil"]:
            out[f"A_selected_dominant_{term}_ratio"] = float((df["A_selected_dominant_term"] == term).mean())

    # How often A-score selection beats each comparator per frame.
    for comp_col in ["sam_iou", "A_only_iou", "C_only_iou", "E_only_iou", "Sil_only_iou"]:
        if comp_col in df.columns:
            out[f"A_score_beats_{comp_col}_ratio"] = float((df["A_score_iou"] > df[comp_col]).mean())
            out[f"A_score_ties_{comp_col}_ratio"] = float(np.isclose(df["A_score_iou"], df[comp_col], atol=1e-12).mean())
            out[f"A_score_loses_to_{comp_col}_ratio"] = float((df["A_score_iou"] < df[comp_col]).mean())

    return out


def summarize_global(seq_df: pd.DataFrame, frame_df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    out["num_sequences"] = int(len(seq_df))
    out["num_frames"] = int(len(frame_df))

    # Macro means over sequences.
    for col in seq_df.columns:
        if col.startswith("mean_"):
            out[f"macro_{col}"] = pd.to_numeric(seq_df[col], errors="coerce").mean()

    # Frame-weighted means over all frames.
    for col in frame_df.columns:
        if col.startswith("rho_") or col.endswith("_iou") or col in [
            "A_gap_to_oracle", "A_delta_vs_sam", "A_gap_to_best_single_term",
            "A_selected_score_A", "A_selected_A", "A_selected_C", "A_selected_E", "A_selected_Sil",
            "A_selected_wA_A", "A_selected_wC_C", "A_selected_wE_E", "A_selected_wSil_Sil",
            "A_selected_A_share", "A_selected_C_share", "A_selected_E_share", "A_selected_Sil_share",
            "A_selected_dominant_share",
        ]:
            out[f"frame_weighted_mean_{col}"] = pd.to_numeric(frame_df[col], errors="coerce").mean()

    if "A_selected_dominant_term" in frame_df.columns:
        for term in ["A", "C", "E", "Sil"]:
            out[f"frame_weighted_A_selected_dominant_{term}_ratio"] = float((frame_df["A_selected_dominant_term"] == term).mean())

    return pd.DataFrame([out])


def parse_sequence_list(args) -> List[str]:
    if args.sequences:
        return [s.strip() for s in args.sequences.split(",") if s.strip()]
    if args.sequence_file:
        path = Path(args.sequence_file)
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.all_sequences:
        return sorted([p.name for p in Path(args.data_root).iterdir() if p.is_dir()])
    return DEFAULT_SEQUENCES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str,
                    default=r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results",
                    help="A-version SAM_results cache root containing <sequence>/json and <sequence>/gt_json")
    ap.add_argument("--output_dir", type=str, default="",
                    help="Output directory. Default: <data_root>/_analysis_A_correlations")
    ap.add_argument("--sequences", type=str, default="",
                    help="Comma-separated sequence list. Default uses the 33 selected sequences.")
    ap.add_argument("--sequence_file", type=str, default="",
                    help="Optional text file containing one sequence name per line.")
    ap.add_argument("--all_sequences", action="store_true",
                    help="Analyze every sequence folder under data_root.")

    # A-version default settings.
    ap.add_argument("--wA", type=float, default=DEFAULT_A_PARAMS["wA"])
    ap.add_argument("--wC", type=float, default=DEFAULT_A_PARAMS["wC"])
    ap.add_argument("--wE", type=float, default=DEFAULT_A_PARAMS["wE"])
    ap.add_argument("--wSil", type=float, default=DEFAULT_A_PARAMS["wSil"])
    ap.add_argument("--t_area", type=float, default=DEFAULT_A_PARAMS["t_area"])
    ap.add_argument("--q_border", type=float, default=DEFAULT_A_PARAMS["q_border"])
    ap.add_argument("--save_candidate_details", action="store_true",
                    help="Save all candidate-level rows. This can be large.")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    output_dir = Path(args.output_dir) if args.output_dir else data_root / "_analysis_A_correlations"
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = normalize_weights(np.asarray([args.wA, args.wC, args.wE, args.wSil], dtype=float))
    params = {
        "wA": float(weights[0]),
        "wC": float(weights[1]),
        "wE": float(weights[2]),
        "wSil": float(weights[3]),
        "t_area": float(args.t_area),
        "q_border": float(args.q_border),
    }

    seqs = parse_sequence_list(args)
    print(f"[INFO] Analyzing {len(seqs)} sequences")
    print(f"[INFO] A params: {params}")

    all_frame_rows = []
    all_candidate_rows = []
    seq_rows = []
    skipped_rows = []

    for seq in seqs:
        seq_dir = data_root / seq
        json_dir = seq_dir / "json"
        gt_dir = seq_dir / "gt_json"
        if not json_dir.exists() or not gt_dir.exists():
            skipped_rows.append({"sequence": seq, "frame": "", "reason": "missing json or gt_json dir"})
            continue

        result_files = sorted(json_dir.glob("*_result.json"))
        frame_rows = []

        for result_path in result_files:
            key = result_path.stem.replace("_result", "")
            gt_path = gt_dir / f"{key}_gt.json"
            if not gt_path.exists():
                skipped_rows.append({"sequence": seq, "frame": key, "reason": "missing gt_json"})
                continue

            frame_row, cand_rows, reason = analyze_frame(
                seq=seq,
                frame_key=key,
                result_path=result_path,
                gt_path=gt_path,
                weights=weights,
                q_border=params["q_border"],
                t_area=params["t_area"],
                save_candidate_details=args.save_candidate_details,
            )
            if frame_row is None:
                skipped_rows.append({"sequence": seq, "frame": key, "reason": reason})
                continue

            frame_rows.append(frame_row)
            all_frame_rows.append(frame_row)
            if args.save_candidate_details:
                all_candidate_rows.extend(cand_rows)

        if frame_rows:
            seq_rows.append(summarize_sequence(seq, frame_rows, params))
        else:
            skipped_rows.append({"sequence": seq, "frame": "", "reason": "no usable frames"})

    frame_df = pd.DataFrame(all_frame_rows)
    seq_df = pd.DataFrame(seq_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    global_df = summarize_global(seq_df, frame_df) if not seq_df.empty and not frame_df.empty else pd.DataFrame()

    frame_path = output_dir / "A_candidate_correlation_per_frame.csv"
    seq_path = output_dir / "A_candidate_correlation_per_sequence.csv"
    global_path = output_dir / "A_candidate_correlation_global_summary.csv"
    skipped_path = output_dir / "A_candidate_correlation_skipped.csv"

    frame_df.to_csv(frame_path, index=False)
    seq_df.to_csv(seq_path, index=False)
    global_df.to_csv(global_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)

    print(f"[DONE] Per-frame analysis saved to: {frame_path}")
    print(f"[DONE] Per-sequence analysis saved to: {seq_path}")
    print(f"[DONE] Global summary saved to: {global_path}")
    print(f"[DONE] Skipped records saved to: {skipped_path}")

    if args.save_candidate_details:
        cand_df = pd.DataFrame(all_candidate_rows)
        cand_path = output_dir / "A_candidate_correlation_candidate_details.csv"
        cand_df.to_csv(cand_path, index=False)
        print(f"[DONE] Candidate details saved to: {cand_path}")

    if not global_df.empty:
        print("\n=== Global summary ===")
        # Print a compact key subset if available.
        key_cols = [
            "num_sequences", "num_frames",
            "frame_weighted_mean_sam_iou",
            "frame_weighted_mean_A_score_iou",
            "frame_weighted_mean_oracle_iou",
            "frame_weighted_mean_A_delta_vs_sam",
            "frame_weighted_mean_A_gap_to_oracle",
            "frame_weighted_mean_rho_score_A_vs_IoU",
            "frame_weighted_mean_rho_A_vs_IoU",
            "frame_weighted_mean_rho_C_vs_IoU",
            "frame_weighted_mean_rho_E_vs_IoU",
            "frame_weighted_mean_rho_Sil_vs_IoU",
            "frame_weighted_mean_rho_sam_score_vs_IoU",
            "frame_weighted_mean_A_selected_A_share",
            "frame_weighted_mean_A_selected_C_share",
            "frame_weighted_mean_A_selected_E_share",
            "frame_weighted_mean_A_selected_Sil_share",
        ]
        key_cols = [c for c in key_cols if c in global_df.columns]
        print(global_df[key_cols].to_string(index=False))


if __name__ == "__main__":
    main()
