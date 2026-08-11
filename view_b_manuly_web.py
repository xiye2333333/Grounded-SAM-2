import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


# =========================================================
# Defaults
# =========================================================

DEFAULT_DATA_ROOT = r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results"
EPS = 1e-8


# =========================================================
# RLE / mask utilities
# =========================================================

def decode_rle(segmentation_rle: Dict[str, Any]) -> np.ndarray:
    """
    Decode COCO-style RLE from JSON.

    Supports:
      1. compressed RLE: counts is str/bytes, requires pycocotools
      2. uncompressed RLE: counts is list
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
                "pycocotools is required to decode compressed COCO RLE. "
                "Install with: pip install pycocotools"
            )
        rle = {"size": [h, w], "counts": counts.encode("utf-8")}
        mask = mask_utils.decode(rle)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask.astype(bool)

    if isinstance(counts, bytes):
        if mask_utils is None:
            raise ImportError("pycocotools is required to decode compressed COCO RLE.")
        rle = {"size": [h, w], "counts": counts}
        mask = mask_utils.decode(rle)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask.astype(bool)

    if isinstance(counts, list):
        # Uncompressed COCO RLE. COCO uses column-major order.
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


def mask_to_pil(mask_bool: np.ndarray) -> Image.Image:
    arr = mask_bool.astype(np.uint8) * 255
    return Image.fromarray(arr)


def overlay_mask_on_image(
    image_pil: Image.Image,
    mask_bool: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.45,
) -> Image.Image:
    image = np.asarray(image_pil.convert("RGB")).astype(np.float32)
    mask = mask_bool.astype(bool)

    if mask.shape[:2] != image.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}.")

    overlay = image.copy()
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * np.asarray(color, dtype=np.float32)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


# =========================================================
# B-version feature functions
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
    """
    Area compatibility score.

    x = observed area ratio
    d = target area ratio

    This preserves the current B-version implementation:
    area is represented as a parabolic compatibility score around t_area.
    """
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

    holes_map = (bg_ff[1:-1, 1:-1] == 1)
    holes_area = float(np.count_nonzero(holes_map))
    hole_ratio = float(np.clip(holes_area / area, 0.0, 1.0)) if area > 0 else 0.0

    sil = solidity * (1.0 - fragmentation) * (1.0 - hole_ratio)
    return float(np.clip(sil, 0.0, 1.0))


def mask_features_old(
    mask_bool: np.ndarray,
    W: int,
    H: int,
    dist_map: np.ndarray,
    d_max: float,
    q_border: float,
    t_area: float,
) -> Dict[str, float]:
    """
    Old observed features and intermediate values:
      area_ratio: raw foreground area / image area
      A: area compatibility with t_area
      C: centeredness
      E_dist: border-distance / interior support
      edge_proximity: 1 - E_dist
      Sil: silhouette
    """
    img_area = W * H
    cx, cy = W / 2, H / 2

    area_px = int(mask_bool.sum())
    if area_px <= 0:
        return {
            "area_ratio": 0.0,
            "A": 0.0,
            "C": 0.0,
            "E_dist": 0.0,
            "edge_proximity": 1.0,
            "Sil": 0.0,
        }

    area_ratio = float(np.clip(area_px / max(img_area, 1), 0.0, 1.0))
    A = area_term_parabola(area_ratio, t_area)

    ys, xs = np.where(mask_bool)
    mx, my = xs.mean(), ys.mean()
    Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
    C = 1.0 - float(np.clip(Dp, 0.0, 1.0))

    q = float(np.quantile(dist_map[mask_bool], q_border))
    E_dist = float(np.clip(q / d_max, 0.0, 1.0)) if d_max > 0 else 0.0
    edge_proximity = float(np.clip(1.0 - E_dist, 0.0, 1.0))

    Sil = compute_silhouette_score_v2(mask_bool)

    return {
        "area_ratio": area_ratio,
        "A": float(A),
        "C": float(C),
        "E_dist": float(E_dist),
        "edge_proximity": edge_proximity,
        "Sil": float(Sil),
    }


def mask_features_b(
    mask_bool: np.ndarray,
    W: int,
    H: int,
    dist_map: np.ndarray,
    d_max: float,
    params: Dict[str, float],
) -> Dict[str, float]:
    """
    Live B-version term scores:
      A_match: area compatibility with t_area
      C_match: closeness to c_target
      E_match: closeness to e_target, where e_target uses edge proximity
      S_match: closeness to s_target
    """
    q_border = float(np.clip(params.get("q_border", 0.25), 0.0, 1.0))
    t_area = float(np.clip(params.get("t_area", 0.25), 1e-6, 1.0))
    c_target = float(np.clip(params.get("c_target", 1.0), 0.0, 1.0))
    e_target = float(np.clip(params.get("e_target", 0.0), 0.0, 1.0))
    s_target = float(np.clip(params.get("s_target", 1.0), 0.0, 1.0))

    old = mask_features_old(mask_bool, W, H, dist_map, d_max, q_border, t_area)

    A_score = float(old["A"])
    C_obs = float(old["C"])
    edge_proximity = float(old["edge_proximity"])
    Sil_obs = float(old["Sil"])

    C_score = 1.0 - abs(c_target - C_obs)
    E_score = 1.0 - abs(e_target - edge_proximity)
    S_score = 1.0 - abs(s_target - Sil_obs)

    out = {
        "A_match": float(np.clip(A_score, 0.0, 1.0)),
        "C_match": float(np.clip(C_score, 0.0, 1.0)),
        "E_match": float(np.clip(E_score, 0.0, 1.0)),
        "S_match": float(np.clip(S_score, 0.0, 1.0)),
        "area_ratio": float(old["area_ratio"]),
        "C_obs": float(old["C"]),
        "E_dist": float(old["E_dist"]),
        "edge_proximity": float(old["edge_proximity"]),
        "Sil_obs": float(old["Sil"]),
    }
    out["score_B"] = out["A_match"] + out["C_match"] + out["E_match"] + out["S_match"]
    return out


def score_masks_b(
    masks: List[np.ndarray],
    W: int,
    H: int,
    dist_map: np.ndarray,
    d_max: float,
    params: Dict[str, float],
) -> Tuple[pd.DataFrame, np.ndarray]:
    rows = [
        mask_features_b(m, W, H, dist_map, d_max, params)
        for m in masks
    ]
    df = pd.DataFrame(rows)
    scores = df["score_B"].to_numpy(dtype=float)
    return df, scores


# =========================================================
# Dataset / result loading
# =========================================================

@st.cache_data(show_spinner=False)
def read_json_cached(path_str: str) -> Dict[str, Any]:
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_image_cached(path_str: str) -> Image.Image:
    return Image.open(path_str).convert("RGB")


def list_available_sequences(data_root: Path) -> List[str]:
    seqs = []
    if not data_root.exists():
        return seqs

    for seq_data_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        seq = seq_data_dir.name
        if seq.startswith("_"):
            continue

        if not (seq_data_dir / "json").exists():
            continue
        if not (seq_data_dir / "gt_json").exists():
            continue

        if not list((seq_data_dir / "json").glob("*_result.json")):
            continue
        if not list((seq_data_dir / "gt_json").glob("*_gt.json")):
            continue

        seqs.append(seq)

    return seqs


def frame_sort_key(x: str):
    return (0, int(x)) if re.fullmatch(r"\d+", x) else (1, x)


def list_frame_ids(seq_data_dir: Path) -> List[str]:
    gt_dir = seq_data_dir / "gt_json"
    json_dir = seq_data_dir / "json"
    if not gt_dir.exists() or not json_dir.exists():
        return []

    ids = []
    for p in gt_dir.glob("*_gt.json"):
        fid = p.stem.replace("_gt", "")
        if (json_dir / f"{fid}_result.json").exists():
            ids.append(fid)

    return sorted(ids, key=frame_sort_key)


def load_gt_frame(seq_data_dir: Path, frame_id: str) -> Dict[str, Any]:
    p = seq_data_dir / "gt_json" / f"{frame_id}_gt.json"
    obj = read_json_cached(str(p))
    gt_mask = decode_rle(obj["segmentation_rle"])

    image_path = obj.get("image_path", "")
    W = int(obj.get("width", gt_mask.shape[1]))
    H = int(obj.get("height", gt_mask.shape[0]))

    return {
        "image_path": image_path,
        "gt_mask": gt_mask,
        "W": W,
        "H": H,
    }


def load_result_frame(seq_data_dir: Path, frame_id: str) -> Dict[str, Any]:
    p = seq_data_dir / "json" / f"{frame_id}_result.json"
    return read_json_cached(str(p))


def load_candidates(result_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    anns = result_json.get("annotations", [])
    candidates = []

    for idx, ann in enumerate(anns):
        mask = decode_rle(ann["segmentation"])
        candidates.append({
            "idx": idx,
            "id": int(ann.get("id", idx)),
            "box_id": int(ann.get("box_id", -1)),
            "rank_in_box": int(ann.get("rank_in_box", -1)),
            "sam_score": float(ann.get("sam_score", np.nan)),
            "mask": mask,
        })

    return candidates


def get_raw_scores(result_json: Dict[str, Any], candidates: List[Dict[str, Any]]) -> np.ndarray:
    raw_scores = result_json.get("raw_scores", None)

    if raw_scores is None or len(raw_scores) != len(candidates):
        raw_scores = [c["sam_score"] for c in candidates]

    raw_scores = np.asarray(raw_scores, dtype=float)
    if raw_scores.size == 0:
        return raw_scores

    if np.all(~np.isfinite(raw_scores)):
        raw_scores = np.zeros_like(raw_scores, dtype=float)

    raw_scores = np.where(np.isfinite(raw_scores), raw_scores, -np.inf)
    return raw_scores


def load_frame_payload(seq_data_dir: Path, frame_id: str):
    gt_data = load_gt_frame(seq_data_dir, frame_id)
    result_json = load_result_frame(seq_data_dir, frame_id)
    raw_image = load_image_cached(gt_data["image_path"])
    gt_mask = gt_data["gt_mask"]

    candidates = load_candidates(result_json)

    valid_candidates = []
    skipped_shape = 0
    for c in candidates:
        if c["mask"].shape == gt_mask.shape:
            valid_candidates.append(c)
        else:
            skipped_shape += 1

    return gt_data, result_json, raw_image, gt_mask, valid_candidates, skipped_shape


def make_candidate_table(
    candidates: List[Dict[str, Any]],
    raw_scores: np.ndarray,
    feature_df: pd.DataFrame,
    gt_mask: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(candidates):
        iou = compute_iou(c["mask"], gt_mask) if c["mask"].shape == gt_mask.shape else np.nan
        feats = feature_df.iloc[i].to_dict()
        row = {
            "idx": c["idx"],
            "id": c["id"],
            "box_id": c["box_id"],
            "rank_in_box": c["rank_in_box"],
            "sam_score": float(raw_scores[i]) if i < len(raw_scores) else np.nan,
            "IoU_vs_GT": float(iou),
        }
        row.update({k: float(v) for k, v in feats.items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["score_B_rank"] = df["score_B"].rank(ascending=False, method="min").astype(int)
        df["sam_score_rank"] = df["sam_score"].rank(ascending=False, method="min").astype(int)
        df["IoU_rank"] = df["IoU_vs_GT"].rank(ascending=False, method="min").astype(int)
        df = df.sort_values("score_B", ascending=False).reset_index(drop=True)
    return df


def score_row_for_idx(
    label: str,
    idx: int,
    candidates: List[Dict[str, Any]],
    raw_scores: np.ndarray,
    feature_df: pd.DataFrame,
    gt_mask: np.ndarray,
) -> Dict[str, Any]:
    mask = candidates[idx]["mask"]
    iou = compute_iou(mask, gt_mask) if mask.shape == gt_mask.shape else np.nan
    feats = feature_df.iloc[idx].to_dict()
    row = {
        "choice": label,
        "idx": int(idx),
        "candidate_id": int(candidates[idx]["id"]),
        "box_id": int(candidates[idx]["box_id"]),
        "rank_in_box": int(candidates[idx]["rank_in_box"]),
        "sam_score": float(raw_scores[idx]) if idx < len(raw_scores) else np.nan,
        "IoU_vs_GT": float(iou),
    }
    row.update({k: float(v) for k, v in feats.items()})
    return row


def format_float(x, digits=4) -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "N/A"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


# =========================================================
# Live sequence-level evaluation
# =========================================================

def basic_stats(values: List[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": np.nan, "min": np.nan, "max": np.nan, "std": np.nan, "n": 0}
    return {
        "mean": float(np.nanmean(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "std": float(np.nanstd(arr)),
        "n": int(arr.size),
    }


def evaluate_sequence_live(seq_data_dir: Path, params: Dict[str, float], candidate_failure_threshold: float = 0.60):
    frame_ids = list_frame_ids(seq_data_dir)

    per_frame = []
    for fid in frame_ids:
        gt_data, result_json, raw_image, gt_mask, candidates, skipped_shape = load_frame_payload(seq_data_dir, fid)
        if not candidates:
            continue

        raw_scores = get_raw_scores(result_json, candidates)
        H, W = gt_mask.shape
        dist_map = compute_distance_transform(H, W)
        d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

        feature_df, score_B = score_masks_b(
            masks=[c["mask"] for c in candidates],
            W=W,
            H=H,
            dist_map=dist_map,
            d_max=d_max,
            params=params,
        )

        candidate_ious = np.asarray([compute_iou(c["mask"], gt_mask) for c in candidates], dtype=float)
        sam_idx = int(np.argmax(raw_scores))
        b_idx = int(np.argmax(score_B))
        oracle_idx = int(np.argmax(candidate_ious))

        sam_iou = float(candidate_ious[sam_idx])
        b_iou = float(candidate_ious[b_idx])
        oracle_iou = float(candidate_ious[oracle_idx])

        per_frame.append({
            "frame": fid,
            "num_candidates": int(len(candidates)),
            "sam_idx": sam_idx,
            "b_idx": b_idx,
            "oracle_idx": oracle_idx,
            "sam_iou": sam_iou,
            "b_iou": b_iou,
            "oracle_iou": oracle_iou,
            "b_delta_vs_sam": b_iou - sam_iou,
            "sam_gap_to_oracle": oracle_iou - sam_iou,
            "b_gap_to_oracle": oracle_iou - b_iou,
            "candidate_set_failed": bool(oracle_iou < candidate_failure_threshold),
            "num_candidates_iou_ge_threshold": int(np.sum(candidate_ious >= candidate_failure_threshold)),
            "mean_candidate_iou": float(np.mean(candidate_ious)),
            "score_B": float(score_B[b_idx]),
            "A_match": float(feature_df.loc[b_idx, "A_match"]),
            "C_match": float(feature_df.loc[b_idx, "C_match"]),
            "E_match": float(feature_df.loc[b_idx, "E_match"]),
            "S_match": float(feature_df.loc[b_idx, "S_match"]),
            "area_ratio": float(feature_df.loc[b_idx, "area_ratio"]),
            "C_obs": float(feature_df.loc[b_idx, "C_obs"]),
            "E_dist": float(feature_df.loc[b_idx, "E_dist"]),
            "edge_proximity": float(feature_df.loc[b_idx, "edge_proximity"]),
            "Sil_obs": float(feature_df.loc[b_idx, "Sil_obs"]),
        })

    df = pd.DataFrame(per_frame)

    if df.empty:
        summary = pd.DataFrame([{
            "n_frames": 0,
            "sam_mean_iou": np.nan,
            "b_mean_iou": np.nan,
            "oracle_mean_iou": np.nan,
            "b_delta_vs_sam": np.nan,
            "b_gap_to_oracle": np.nan,
            "candidate_set_failure_ratio": np.nan,
        }])
        return summary, df

    summary_row = {
        "n_frames": int(len(df)),
        "sam_mean_iou": float(df["sam_iou"].mean()),
        "b_mean_iou": float(df["b_iou"].mean()),
        "oracle_mean_iou": float(df["oracle_iou"].mean()),
        "b_delta_vs_sam": float((df["b_iou"] - df["sam_iou"]).mean()),
        "sam_gap_to_oracle": float((df["oracle_iou"] - df["sam_iou"]).mean()),
        "b_gap_to_oracle": float((df["oracle_iou"] - df["b_iou"]).mean()),
        "candidate_failure_threshold": float(candidate_failure_threshold),
        "candidate_set_failure_ratio": float(df["candidate_set_failed"].mean()),
        "mean_num_candidates": float(df["num_candidates"].mean()),
        "mean_num_candidates_iou_ge_threshold": float(df["num_candidates_iou_ge_threshold"].mean()),
        "mean_score_B": float(df["score_B"].mean()),
        "std_score_B": float(df["score_B"].std(ddof=0)),
        "mean_A_match": float(df["A_match"].mean()),
        "mean_C_match": float(df["C_match"].mean()),
        "mean_E_match": float(df["E_match"].mean()),
        "mean_S_match": float(df["S_match"].mean()),
        "std_A_match": float(df["A_match"].std(ddof=0)),
        "std_C_match": float(df["C_match"].std(ddof=0)),
        "std_E_match": float(df["E_match"].std(ddof=0)),
        "std_S_match": float(df["S_match"].std(ddof=0)),
    }
    summary = pd.DataFrame([summary_row])
    return summary, df


# =========================================================
# Streamlit app
# =========================================================

st.set_page_config(
    page_title="Manual B-version ACES DAVIS Viewer",
    layout="wide",
)

st.title("Manual B-version ACES DAVIS Viewer")
st.caption(
    "Manual tuning viewer for DAVIS candidate masks. "
    "This app does not read optimized B result files or saved evaluation CSVs; "
    "it recomputes B scores live from the sliders and the cached candidate masks."
)

with st.sidebar:
    st.header("Data source")

    data_root_str = st.text_input(
        "Candidate cache root",
        value=DEFAULT_DATA_ROOT,
        help="Original SAM_results root containing json/ and gt_json/ for each DAVIS sequence.",
    )
    data_root = Path(data_root_str)

    seqs = list_available_sequences(data_root)
    if not seqs:
        st.error("No DAVIS sequence folders found. Check data_root.")
        st.stop()

    st.header("Sequence / frame")
    sequence = st.selectbox("Sequence", options=seqs, index=0)
    seq_data_dir = data_root / sequence

    frame_ids = list_frame_ids(seq_data_dir)
    if not frame_ids:
        st.error(f"No valid frames found for sequence: {sequence}")
        st.stop()

    frame_id = st.selectbox("Frame", options=frame_ids, index=0)

    st.divider()
    st.header("Manual B assumptions")

    st.markdown("**B score:** `A_match + C_match + E_match + S_match`")

    preset = st.selectbox(
        "Starting preset",
        options=["B default", "Typical DAVIS GT-mean approx", "Custom"],
        index=0,
        help=(
            "The preset only provides slider initial values. "
            "Move the sliders to manually tune the current B assumptions."
        ),
    )

    if preset == "Typical DAVIS GT-mean approx":
        default_q, default_t, default_c, default_e, default_s = 0.25, 0.085, 0.806, 0.399, 0.624
    else:
        default_q, default_t, default_c, default_e, default_s = 0.25, 0.25, 1.0, 0.0, 1.0

    q_border = st.slider(
        "q_border",
        min_value=0.00,
        max_value=1.00,
        value=float(default_q),
        step=0.01,
        help="Quantile used to observe old E_dist border-distance support.",
    )
    t_area = st.slider(
        "t_area",
        min_value=0.001,
        max_value=0.95,
        value=float(default_t),
        step=0.001,
        format="%.3f",
        help="Expected target area ratio. A_match is a parabolic compatibility around this value.",
    )
    c_target = st.slider(
        "c_target",
        min_value=0.00,
        max_value=1.00,
        value=float(default_c),
        step=0.01,
        help="Expected centeredness. 1 means centered; 0 means far from image center.",
    )
    e_target = st.slider(
        "e_target",
        min_value=0.00,
        max_value=1.00,
        value=float(default_e),
        step=0.01,
        help="Expected edge proximity. 1 means close to image edge; 0 means far from image edge.",
    )
    s_target = st.slider(
        "s_target",
        min_value=0.00,
        max_value=1.00,
        value=float(default_s),
        step=0.01,
        help="Expected silhouette score. 1 means ideal silhouette.",
    )

    params = {
        "q_border": q_border,
        "t_area": t_area,
        "c_target": c_target,
        "e_target": e_target,
        "s_target": s_target,
    }

    st.divider()
    st.header("Display")
    show_overlay = st.checkbox("Show masks as overlays", value=True)
    overlay_alpha = st.slider("Overlay alpha", 0.05, 0.95, 0.45, 0.05)

    show_candidate_table = st.checkbox("Show candidate table", value=True)
    show_candidate_gallery = st.checkbox("Show candidate image gallery", value=True)
    show_term_chart = st.checkbox("Show selected-mask term chart", value=True)

    if show_candidate_gallery:
        gallery_sort_by = st.selectbox(
            "Candidate gallery sort by",
            options=["score_B", "IoU_vs_GT", "sam_score", "id"],
            index=0,
        )
        gallery_cols_per_row = st.slider("Candidate images per row", 2, 6, 4, 1)
        gallery_max_items = st.number_input(
            "Max candidate images to show",
            min_value=1,
            max_value=200,
            value=60,
            step=1,
        )
    else:
        gallery_sort_by = "score_B"
        gallery_cols_per_row = 4
        gallery_max_items = 60

    st.divider()
    st.header("Sequence-level live evaluation")
    candidate_failure_threshold = st.slider(
        "Candidate-set failure IoU threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.01,
        help="A frame is candidate-set failure if the best candidate IoU is below this value.",
    )
    run_sequence_eval = st.button("Recompute current sequence with current parameters")


# Load selected frame
gt_data, result_json, raw_image, gt_mask, candidates, skipped_shape = load_frame_payload(seq_data_dir, frame_id)

if skipped_shape:
    st.warning(f"Skipped {skipped_shape} candidates because their shapes do not match GT.")

if not candidates:
    st.error("No valid candidate masks in this frame.")
    st.stop()

H, W = gt_mask.shape
dist_map = compute_distance_transform(H, W)
d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

raw_scores = get_raw_scores(result_json, candidates)
feature_df, score_B = score_masks_b(
    masks=[c["mask"] for c in candidates],
    W=W,
    H=H,
    dist_map=dist_map,
    d_max=d_max,
    params=params,
)

candidate_df = make_candidate_table(candidates, raw_scores, feature_df, gt_mask)

sam_idx = int(np.argmax(raw_scores))
b_idx = int(np.argmax(score_B))
oracle_idx = int(candidate_df.loc[candidate_df["IoU_vs_GT"].idxmax(), "idx"])

sam_row = score_row_for_idx("SAM raw-score choice", sam_idx, candidates, raw_scores, feature_df, gt_mask)
b_row = score_row_for_idx("Manual B score choice", b_idx, candidates, raw_scores, feature_df, gt_mask)
oracle_row = score_row_for_idx("Oracle best candidate", oracle_idx, candidates, raw_scores, feature_df, gt_mask)


# =========================================================
# Sequence / parameter summary
# =========================================================

st.subheader(f"Sequence: {sequence} | Frame: {frame_id}")

param_cols = st.columns(5)
param_cols[0].metric("q_border", format_float(params["q_border"]))
param_cols[1].metric("t_area", format_float(params["t_area"], digits=3))
param_cols[2].metric("c_target", format_float(params["c_target"]))
param_cols[3].metric("e_target", format_float(params["e_target"]))
param_cols[4].metric("s_target", format_float(params["s_target"]))

metric_cols = st.columns(6)
metric_cols[0].metric("Num candidates", len(candidates))
metric_cols[1].metric("SAM IoU", format_float(sam_row["IoU_vs_GT"]))
metric_cols[2].metric("Manual B IoU", format_float(b_row["IoU_vs_GT"]))
metric_cols[3].metric("Oracle IoU", format_float(oracle_row["IoU_vs_GT"]))
metric_cols[4].metric("B - SAM IoU", format_float(b_row["IoU_vs_GT"] - sam_row["IoU_vs_GT"]))
metric_cols[5].metric("B gap to oracle", format_float(oracle_row["IoU_vs_GT"] - b_row["IoU_vs_GT"]))

st.code(
    f"score_B = {b_row['A_match']:.4f} + {b_row['C_match']:.4f} + "
    f"{b_row['E_match']:.4f} + {b_row['S_match']:.4f} = {b_row['score_B']:.4f}",
    language="text",
)


# =========================================================
# Main image display
# =========================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("**Raw image**")
    st.image(raw_image, use_container_width=True)
    st.caption(str(gt_data["image_path"]))

with col2:
    st.markdown("**GT mask**")
    if show_overlay:
        st.image(
            overlay_mask_on_image(raw_image, gt_mask, color=(0, 255, 0), alpha=overlay_alpha),
            use_container_width=True,
        )
    else:
        st.image(mask_to_pil(gt_mask), use_container_width=True)
    st.caption("Green overlay = GT")

with col3:
    st.markdown("**SAM raw-score choice**")
    sam_mask = candidates[sam_idx]["mask"]
    if show_overlay:
        st.image(
            overlay_mask_on_image(raw_image, sam_mask, color=(255, 0, 0), alpha=overlay_alpha),
            use_container_width=True,
        )
    else:
        st.image(mask_to_pil(sam_mask), use_container_width=True)
    st.write(f"idx/id: `{sam_idx}` / `{sam_row['candidate_id']}`")
    st.write(f"SAM score: `{format_float(sam_row['sam_score'])}`")
    st.write(f"score_B: `{format_float(sam_row['score_B'])}`")
    st.write(f"IoU: `{format_float(sam_row['IoU_vs_GT'])}`")

with col4:
    st.markdown("**Manual B score choice**")
    b_mask = candidates[b_idx]["mask"]
    if show_overlay:
        st.image(
            overlay_mask_on_image(raw_image, b_mask, color=(0, 0, 255), alpha=overlay_alpha),
            use_container_width=True,
        )
    else:
        st.image(mask_to_pil(b_mask), use_container_width=True)
    st.write(f"idx/id: `{b_idx}` / `{b_row['candidate_id']}`")
    st.write(f"SAM score: `{format_float(b_row['sam_score'])}`")
    st.write(f"score_B: `{format_float(b_row['score_B'])}`")
    st.write(f"IoU: `{format_float(b_row['IoU_vs_GT'])}`")

with col5:
    st.markdown("**Oracle best candidate**")
    oracle_mask = candidates[oracle_idx]["mask"]
    if show_overlay:
        st.image(
            overlay_mask_on_image(raw_image, oracle_mask, color=(0, 255, 255), alpha=overlay_alpha),
            use_container_width=True,
        )
    else:
        st.image(mask_to_pil(oracle_mask), use_container_width=True)
    st.write(f"idx/id: `{oracle_idx}` / `{oracle_row['candidate_id']}`")
    st.write(f"SAM score: `{format_float(oracle_row['sam_score'])}`")
    st.write(f"score_B: `{format_float(oracle_row['score_B'])}`")
    st.write(f"IoU: `{format_float(oracle_row['IoU_vs_GT'])}`")


# =========================================================
# Term tables and chart
# =========================================================

st.subheader("Live score_B and term values")

comparison_df = pd.DataFrame([sam_row, b_row, oracle_row])
for term in ["A_match", "C_match", "E_match", "S_match"]:
    comparison_df[f"{term}_share"] = comparison_df[term] / comparison_df["score_B"].replace(0, np.nan)

display_cols = [
    "choice", "idx", "candidate_id", "sam_score", "score_B",
    "A_match", "C_match", "E_match", "S_match",
    "A_match_share", "C_match_share", "E_match_share", "S_match_share",
    "area_ratio", "C_obs", "E_dist", "edge_proximity", "Sil_obs",
    "IoU_vs_GT",
]
display_cols = [c for c in display_cols if c in comparison_df.columns]
st.dataframe(comparison_df[display_cols], use_container_width=True, hide_index=True)

if show_term_chart:
    st.markdown("**Term values for SAM / Manual B / Oracle**")
    chart_df = comparison_df[["choice", "A_match", "C_match", "E_match", "S_match"]].set_index("choice")
    st.bar_chart(chart_df.T)


# =========================================================
# Sequence-level live evaluation
# =========================================================

if run_sequence_eval:
    with st.spinner("Recomputing current sequence using current manual parameters..."):
        summary_df, per_frame_eval_df = evaluate_sequence_live(
            seq_data_dir=seq_data_dir,
            params=params,
            candidate_failure_threshold=candidate_failure_threshold,
        )

    st.subheader("Live sequence-level evaluation")
    st.write(
        "This table is computed live from json/gt_json and the current sliders. "
        "It does not read saved B evaluation summaries."
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.download_button(
        label="Download live sequence summary CSV",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{sequence}_manual_B_live_sequence_summary.csv",
        mime="text/csv",
    )

    with st.expander("Per-frame live evaluation", expanded=False):
        st.dataframe(per_frame_eval_df, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download live per-frame CSV",
            data=per_frame_eval_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{sequence}_manual_B_live_per_frame.csv",
            mime="text/csv",
        )


# =========================================================
# All candidate table
# =========================================================

if show_candidate_table:
    st.subheader("All candidates in this frame")

    table_df = candidate_df.copy()

    preferred_cols = [
        "idx", "id", "box_id", "rank_in_box",
        "sam_score", "sam_score_rank",
        "score_B", "score_B_rank",
        "A_match", "C_match", "E_match", "S_match",
        "area_ratio", "C_obs", "E_dist", "edge_proximity", "Sil_obs",
        "IoU_vs_GT", "IoU_rank",
    ]
    preferred_cols = [c for c in preferred_cols if c in table_df.columns]

    st.dataframe(table_df[preferred_cols], use_container_width=True, hide_index=True)

    st.download_button(
        label="Download this frame's live candidate table",
        data=table_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{sequence}_{frame_id}_manual_B_candidate_scores.csv",
        mime="text/csv",
    )


# =========================================================
# All candidate image gallery
# =========================================================

if show_candidate_gallery:
    st.subheader("All candidate mask images")

    gallery_df = candidate_df.copy()

    if gallery_sort_by not in gallery_df.columns:
        gallery_sort_by = "score_B"

    ascending = gallery_sort_by == "id"
    gallery_df = gallery_df.sort_values(gallery_sort_by, ascending=ascending).reset_index(drop=True)

    max_items = int(min(gallery_max_items, len(gallery_df)))
    gallery_df = gallery_df.head(max_items)

    st.caption(
        f"Showing {len(gallery_df)} / {len(candidate_df)} candidates, sorted by `{gallery_sort_by}`."
    )

    for start in range(0, len(gallery_df), int(gallery_cols_per_row)):
        cols = st.columns(int(gallery_cols_per_row))
        chunk = gallery_df.iloc[start:start + int(gallery_cols_per_row)]

        for col, (_, row) in zip(cols, chunk.iterrows()):
            idx = int(row["idx"])
            mask = candidates[idx]["mask"]

            tags = []
            if idx == sam_idx:
                tags.append("SAM")
            if idx == b_idx:
                tags.append("Manual B")
            if idx == oracle_idx:
                tags.append("Oracle")

            tag_text = f" [{' | '.join(tags)}]" if tags else ""

            if show_overlay:
                if idx == b_idx:
                    color = (0, 0, 255)
                elif idx == sam_idx:
                    color = (255, 0, 0)
                elif idx == oracle_idx:
                    color = (0, 255, 255)
                else:
                    color = (255, 255, 0)
                img_to_show = overlay_mask_on_image(raw_image, mask, color=color, alpha=overlay_alpha)
            else:
                img_to_show = mask_to_pil(mask)

            with col:
                st.markdown(f"**idx {idx} / id {int(row['id'])}{tag_text}**")
                st.image(img_to_show, use_container_width=True)
                st.caption(
                    f"IoU={format_float(row['IoU_vs_GT'])} | "
                    f"score_B={format_float(row['score_B'])} | "
                    f"SAM={format_float(row['sam_score'])}"
                )
                st.caption(
                    f"A={format_float(row['A_match'])}, "
                    f"C={format_float(row['C_match'])}, "
                    f"E={format_float(row['E_match'])}, "
                    f"S={format_float(row['S_match'])}"
                )

    if len(candidate_df) > len(gallery_df):
        st.info(
            f"Only the first {len(gallery_df)} candidates are displayed. "
            "Increase 'Max candidate images to show' in the sidebar to see more."
        )
