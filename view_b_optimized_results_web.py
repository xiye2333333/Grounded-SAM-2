import argparse
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
DEFAULT_B_ROOT = r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results_B_GT_mean"
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
) -> np.ndarray:
    """
    Old observed features:
      A: area compatibility
      C: centeredness
      E_dist: border-distance / interior support
      Sil: silhouette
    """
    img_area = W * H
    cx, cy = W / 2, H / 2

    area_px = int(mask_bool.sum())
    if area_px <= 0:
        return np.asarray([0.0, 0.0, 0.0, 0.0], dtype=float)

    A_raw = area_px / img_area
    A = area_term_parabola(A_raw, t_area)

    ys, xs = np.where(mask_bool)
    mx, my = xs.mean(), ys.mean()
    Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
    C = 1.0 - float(np.clip(Dp, 0.0, 1.0))

    q = float(np.quantile(dist_map[mask_bool], q_border))
    E_dist = float(np.clip(q / d_max, 0.0, 1.0))

    Sil = compute_silhouette_score_v2(mask_bool)

    return np.asarray([A, C, E_dist, Sil], dtype=float)


def mask_features_b(
    mask_bool: np.ndarray,
    W: int,
    H: int,
    dist_map: np.ndarray,
    d_max: float,
    params: Dict[str, float],
) -> np.ndarray:
    """
    B-version term scores:
      A_match: existing area compatibility with t_area
      C_match: closeness to c_target
      E_match: closeness to e_target, where e_target uses edge proximity
      S_match: closeness to s_target

    Important:
      Old E is border-distance/interior support.
      B's e_target is edge proximity.
      Therefore: edge_proximity = 1 - E_dist.
    """
    q_border = float(np.clip(params.get("q_border", 0.25), 0.0, 1.0))
    t_area = float(np.clip(params.get("t_area", 0.25), 0.0, 1.0))
    c_target = float(np.clip(params.get("c_target", 1.0), 0.0, 1.0))
    e_target = float(np.clip(params.get("e_target", 0.0), 0.0, 1.0))
    s_target = float(np.clip(params.get("s_target", 1.0), 0.0, 1.0))

    old = mask_features_old(mask_bool, W, H, dist_map, d_max, q_border, t_area)
    A_score = float(old[0])
    C_obs = float(old[1])
    E_dist = float(old[2])
    edge_proximity = 1.0 - E_dist
    Sil_obs = float(old[3])

    C_score = 1.0 - abs(c_target - C_obs)
    E_score = 1.0 - abs(e_target - edge_proximity)
    S_score = 1.0 - abs(s_target - Sil_obs)

    return np.asarray([
        np.clip(A_score, 0.0, 1.0),
        np.clip(C_score, 0.0, 1.0),
        np.clip(E_score, 0.0, 1.0),
        np.clip(S_score, 0.0, 1.0),
    ], dtype=float)


def score_masks_b(
    masks: List[np.ndarray],
    W: int,
    H: int,
    dist_map: np.ndarray,
    d_max: float,
    params: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    X_b = np.stack([
        mask_features_b(m, W, H, dist_map, d_max, params)
        for m in masks
    ], axis=0)
    scores = X_b.sum(axis=1)
    return X_b, scores


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


def load_optimized_params(seq_b_dir: Path) -> Optional[Dict[str, float]]:
    p = seq_b_dir / "optimized_params_B.json"
    if not p.exists():
        return None

    obj = read_json_cached(str(p))
    if obj.get("skipped", False):
        return None

    opt = obj.get("optimized", {})
    required = ["q_border", "t_area", "c_target", "e_target", "s_target"]
    if not all(k in opt for k in required):
        return None

    return {
        "q_border": float(opt["q_border"]),
        "t_area": float(opt["t_area"]),
        "c_target": float(opt["c_target"]),
        "e_target": float(opt["e_target"]),
        "s_target": float(opt["s_target"]),
    }


def list_available_sequences(data_root: Path, b_root: Path) -> List[str]:
    seqs = []
    if not b_root.exists() or not data_root.exists():
        return seqs

    for seq_b_dir in sorted([p for p in b_root.iterdir() if p.is_dir()]):
        seq = seq_b_dir.name
        if seq.startswith("_"):
            continue

        params = load_optimized_params(seq_b_dir)
        if params is None:
            continue

        seq_data_dir = data_root / seq
        if not (seq_data_dir / "json").exists():
            continue
        if not (seq_data_dir / "gt_json").exists():
            continue

        # Need at least one result json and one GT json.
        if not list((seq_data_dir / "json").glob("*_result.json")):
            continue
        if not list((seq_data_dir / "gt_json").glob("*_gt.json")):
            continue

        seqs.append(seq)

    return seqs


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

    def sort_key(x):
        # Numeric frame IDs sort numerically; other IDs sort lexicographically.
        return (0, int(x)) if re.fullmatch(r"\d+", x) else (1, x)

    return sorted(ids, key=sort_key)


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

    # If all scores are nan, fallback to zeros.
    if np.all(~np.isfinite(raw_scores)):
        raw_scores = np.zeros_like(raw_scores, dtype=float)

    # For argmax safety: nan should not win.
    raw_scores = np.where(np.isfinite(raw_scores), raw_scores, -np.inf)
    return raw_scores


def load_eval_per_frame_if_exists(seq_b_dir: Path) -> Optional[pd.DataFrame]:
    p = seq_b_dir / "eval_per_frame_B_optimized.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def row_from_eval_csv(eval_df: Optional[pd.DataFrame], frame_id: str) -> Optional[pd.Series]:
    if eval_df is None or eval_df.empty or "frame" not in eval_df.columns:
        return None

    # CSV frame may be numeric if pandas guessed int. Compare as string.
    tmp = eval_df.copy()
    tmp["_frame_str"] = tmp["frame"].astype(str).str.zfill(len(frame_id)) if frame_id.isdigit() else tmp["frame"].astype(str)
    hit = tmp[tmp["_frame_str"] == frame_id]
    if hit.empty:
        # Try non-zfilled.
        hit = tmp[tmp["frame"].astype(str) == frame_id]
    if hit.empty:
        return None

    return hit.iloc[0]


def make_candidate_table(
    candidates: List[Dict[str, Any]],
    raw_scores: np.ndarray,
    X_b: np.ndarray,
    score_B: np.ndarray,
    gt_mask: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(candidates):
        iou = compute_iou(c["mask"], gt_mask) if c["mask"].shape == gt_mask.shape else np.nan
        rows.append({
            "idx": c["idx"],
            "id": c["id"],
            "box_id": c["box_id"],
            "rank_in_box": c["rank_in_box"],
            "sam_score": float(raw_scores[i]) if i < len(raw_scores) else np.nan,
            "score_B": float(score_B[i]),
            "A_match": float(X_b[i, 0]),
            "C_match": float(X_b[i, 1]),
            "E_match": float(X_b[i, 2]),
            "S_match": float(X_b[i, 3]),
            "IoU_vs_GT": float(iou),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score_B", ascending=False).reset_index(drop=True)
    return df


def score_row_for_idx(
    label: str,
    idx: int,
    candidates: List[Dict[str, Any]],
    raw_scores: np.ndarray,
    X_b: np.ndarray,
    score_B: np.ndarray,
    gt_mask: np.ndarray,
) -> Dict[str, Any]:
    mask = candidates[idx]["mask"]
    iou = compute_iou(mask, gt_mask) if mask.shape == gt_mask.shape else np.nan
    return {
        "choice": label,
        "idx": int(idx),
        "candidate_id": int(candidates[idx]["id"]),
        "box_id": int(candidates[idx]["box_id"]),
        "rank_in_box": int(candidates[idx]["rank_in_box"]),
        "sam_score": float(raw_scores[idx]) if idx < len(raw_scores) else np.nan,
        "score_B": float(score_B[idx]),
        "A_match": float(X_b[idx, 0]),
        "C_match": float(X_b[idx, 1]),
        "E_match": float(X_b[idx, 2]),
        "S_match": float(X_b[idx, 3]),
        "IoU_vs_GT": float(iou),
    }


def format_float(x, digits=4) -> str:
    try:
        if x is None or not np.isfinite(float(x)):
            return "N/A"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


# =========================================================
# Streamlit app
# =========================================================

st.set_page_config(
    page_title="B-version ACES Sequence Viewer",
    layout="wide",
)

st.title("B-version ACES Sequence Viewer")
st.caption(
    "Inspect sequence-specific optimized B parameters, SAM-selected masks, and B-score-selected masks. "
    "Incomplete sequences without optimized B parameters are skipped automatically."
)

with st.sidebar:
    st.header("Paths")

    data_root_str = st.text_input(
        "Candidate cache root",
        value=DEFAULT_DATA_ROOT,
        help="Usually the original SAM_results root containing json/, gt_json/, and masks/ for each sequence.",
    )

    b_root_str = st.text_input(
        "B-version result root",
        value=DEFAULT_B_ROOT,
        help="Usually SAM_results_B, containing optimized_params_B.json and eval_per_frame_B_optimized.csv.",
    )

    data_root = Path(data_root_str)
    b_root = Path(b_root_str)

    seqs = list_available_sequences(data_root, b_root)

    if not seqs:
        st.error("No completed B-version sequences found. Check data_root and b_root.")
        st.stop()

    st.header("Selection")

    sequence = st.selectbox(
        "Sequence",
        options=seqs,
        index=0,
    )

    seq_data_dir = data_root / sequence
    seq_b_dir = b_root / sequence
    params = load_optimized_params(seq_b_dir)

    if params is None:
        st.error(f"Missing valid optimized B params for sequence: {sequence}")
        st.stop()

    frame_ids = list_frame_ids(seq_data_dir)
    if not frame_ids:
        st.error(f"No valid frames found for sequence: {sequence}")
        st.stop()

    frame_id = st.selectbox(
        "Frame",
        options=frame_ids,
        index=0,
    )

    st.divider()

    st.header("Display")
    show_overlay = st.checkbox("Show masks as overlays", value=True)
    overlay_alpha = st.slider("Overlay alpha", 0.05, 0.95, 0.45, 0.05)
    show_candidate_table = st.checkbox("Show all candidate table", value=True)
    show_candidate_gallery = st.checkbox("Show all candidate mask images", value=True)
    show_term_chart = st.checkbox("Show selected-mask term chart", value=True)

    if show_candidate_gallery:
        gallery_sort_by = st.selectbox(
            "Candidate gallery sort by",
            options=["score_B", "IoU_vs_GT", "sam_score", "id"],
            index=0,
            help="score_B is usually the most useful order for diagnosing B-version selection."
        )
        gallery_cols_per_row = st.slider("Candidate images per row", 2, 6, 4, 1)
        gallery_max_items = st.number_input(
            "Max candidate images to show",
            min_value=1,
            max_value=200,
            value=60,
            step=1,
            help="This only limits display count. The score table still contains all candidates."
        )
    else:
        gallery_sort_by = "score_B"
        gallery_cols_per_row = 4
        gallery_max_items = 60

    st.divider()

    st.header("What-if weighted diagnosis")
    enable_weighted = st.checkbox(
        "Enable temporary term weights",
        value=False,
        help=(
            "This does not change the optimized B parameters. "
            "It only tests whether a weighted sum would select a different mask."
        ),
    )

    if enable_weighted:
        wA = st.slider("wA", 0.0, 3.0, 1.0, 0.05)
        wC = st.slider("wC", 0.0, 3.0, 1.0, 0.05)
        wE = st.slider("wE", 0.0, 3.0, 1.0, 0.05)
        wS = st.slider("wS", 0.0, 3.0, 1.0, 0.05)
    else:
        wA = wC = wE = wS = 1.0


# Load data for selected frame
gt_data = load_gt_frame(seq_data_dir, frame_id)
result_json = load_result_frame(seq_data_dir, frame_id)
raw_image = load_image_cached(gt_data["image_path"])
gt_mask = gt_data["gt_mask"]

H, W = gt_mask.shape
dist_map = compute_distance_transform(H, W)
d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

candidates = load_candidates(result_json)
if not candidates:
    st.error("No candidate masks in this frame.")
    st.stop()

# Filter candidates with valid shape.
valid_candidates = []
skipped_shape = 0
for c in candidates:
    if c["mask"].shape == gt_mask.shape:
        valid_candidates.append(c)
    else:
        skipped_shape += 1

if not valid_candidates:
    st.error("No candidate mask has the same shape as the GT mask.")
    st.stop()

if skipped_shape:
    st.warning(f"Skipped {skipped_shape} candidates because their shapes do not match GT.")

candidates = valid_candidates
raw_scores = get_raw_scores(result_json, candidates)

X_b, score_B = score_masks_b(
    masks=[c["mask"] for c in candidates],
    W=W,
    H=H,
    dist_map=dist_map,
    d_max=d_max,
    params=params,
)

sam_idx = int(np.argmax(raw_scores))
b_idx = int(np.argmax(score_B))

weights = np.asarray([wA, wC, wE, wS], dtype=float)
weighted_score = X_b @ weights
weighted_idx = int(np.argmax(weighted_score))

sam_row = score_row_for_idx("SAM raw-score choice", sam_idx, candidates, raw_scores, X_b, score_B, gt_mask)
b_row = score_row_for_idx("B score choice", b_idx, candidates, raw_scores, X_b, score_B, gt_mask)
weighted_row = score_row_for_idx("Weighted what-if choice", weighted_idx, candidates, raw_scores, X_b, weighted_score, gt_mask)

candidate_df = make_candidate_table(candidates, raw_scores, X_b, score_B, gt_mask)

eval_df = load_eval_per_frame_if_exists(seq_b_dir)
eval_row = row_from_eval_csv(eval_df, frame_id)

# =========================================================
# Sequence / parameter summary
# =========================================================

st.subheader(f"Sequence: {sequence} | Frame: {frame_id}")

param_cols = st.columns(5)
param_cols[0].metric("q_border", format_float(params["q_border"]))
param_cols[1].metric("t_area", format_float(params["t_area"]))
param_cols[2].metric("c_target", format_float(params["c_target"]))
param_cols[3].metric("e_target", format_float(params["e_target"]))
param_cols[4].metric("s_target", format_float(params["s_target"]))

metric_cols = st.columns(5)
metric_cols[0].metric("Num candidates", len(candidates))
metric_cols[1].metric("SAM IoU", format_float(sam_row["IoU_vs_GT"]))
metric_cols[2].metric("B IoU", format_float(b_row["IoU_vs_GT"]))
metric_cols[3].metric("B - SAM IoU", format_float(b_row["IoU_vs_GT"] - sam_row["IoU_vs_GT"]))
metric_cols[4].metric("B score gap", format_float(float(score_B[b_idx] - score_B[sam_idx])))

if eval_row is not None:
    with st.expander("Saved eval_per_frame_B_optimized.csv row", expanded=False):
        row_dict = eval_row.drop(labels=[c for c in ["_frame_str"] if c in eval_row.index]).to_dict()
        st.dataframe(pd.DataFrame([row_dict]), use_container_width=True, hide_index=True)


# =========================================================
# Image display
# =========================================================

col1, col2, col3, col4 = st.columns(4)

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

    st.write(f"candidate idx/id: `{sam_idx}` / `{sam_row['candidate_id']}`")
    st.write(f"SAM raw score: `{format_float(sam_row['sam_score'])}`")
    st.write(f"score_B: `{format_float(sam_row['score_B'])}`")
    st.write(f"IoU vs GT: `{format_float(sam_row['IoU_vs_GT'])}`")

with col4:
    st.markdown("**B score choice**")
    b_mask = candidates[b_idx]["mask"]
    if show_overlay:
        st.image(
            overlay_mask_on_image(raw_image, b_mask, color=(0, 0, 255), alpha=overlay_alpha),
            use_container_width=True,
        )
    else:
        st.image(mask_to_pil(b_mask), use_container_width=True)

    st.write(f"candidate idx/id: `{b_idx}` / `{b_row['candidate_id']}`")
    st.write(f"SAM raw score: `{format_float(b_row['sam_score'])}`")
    st.write(f"score_B: `{format_float(b_row['score_B'])}`")
    st.write(f"IoU vs GT: `{format_float(b_row['IoU_vs_GT'])}`")


# =========================================================
# Term tables
# =========================================================

st.subheader("Score-B and term values")

comparison_df = pd.DataFrame([sam_row, b_row])
if enable_weighted:
    weighted_row_display = weighted_row.copy()
    weighted_row_display["score_B"] = float(weighted_score[weighted_idx])
    comparison_df = pd.concat([comparison_df, pd.DataFrame([weighted_row_display])], ignore_index=True)

# Add diagnostic columns showing how much each term contributes to unweighted score_B.
for term in ["A_match", "C_match", "E_match", "S_match"]:
    comparison_df[f"{term}_share"] = comparison_df[term] / comparison_df["score_B"].replace(0, np.nan)

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True,
)

if show_term_chart:
    st.markdown("**Term values for SAM choice vs B choice**")
    chart_df = comparison_df[comparison_df["choice"].isin(["SAM raw-score choice", "B score choice"])][
        ["choice", "A_match", "C_match", "E_match", "S_match"]
    ].set_index("choice")
    st.bar_chart(chart_df.T)


# =========================================================
# What-if weighted diagnosis
# =========================================================

if enable_weighted:
    st.subheader("Temporary weighted-sum diagnosis")

    st.write(
        "This section keeps the sequence-specific optimized B parameters fixed, "
        "but changes the final selection rule from unweighted sum to a temporary weighted sum."
    )

    st.code(
        f"weighted_score = {wA:.2f}*A_match + {wC:.2f}*C_match + {wE:.2f}*E_match + {wS:.2f}*S_match",
        language="text",
    )

    if weighted_idx == b_idx:
        st.info("The temporary weighted sum selects the same mask as the original unweighted B score.")
    else:
        st.warning(
            f"The temporary weighted sum selects a different mask: "
            f"original B idx={b_idx}, weighted idx={weighted_idx}."
        )

    wcol1, wcol2 = st.columns(2)

    with wcol1:
        st.markdown("**Original B choice**")
        st.image(
            overlay_mask_on_image(raw_image, candidates[b_idx]["mask"], color=(0, 0, 255), alpha=overlay_alpha)
            if show_overlay else mask_to_pil(candidates[b_idx]["mask"]),
            use_container_width=True,
        )
        st.write(f"IoU vs GT: `{format_float(b_row['IoU_vs_GT'])}`")
        st.write(f"unweighted score_B: `{format_float(score_B[b_idx])}`")
        st.write(f"weighted score: `{format_float(weighted_score[b_idx])}`")

    with wcol2:
        st.markdown("**Weighted what-if choice**")
        st.image(
            overlay_mask_on_image(raw_image, candidates[weighted_idx]["mask"], color=(255, 255, 0), alpha=overlay_alpha)
            if show_overlay else mask_to_pil(candidates[weighted_idx]["mask"]),
            use_container_width=True,
        )
        st.write(f"IoU vs GT: `{format_float(weighted_row['IoU_vs_GT'])}`")
        st.write(f"unweighted score_B: `{format_float(score_B[weighted_idx])}`")
        st.write(f"weighted score: `{format_float(weighted_score[weighted_idx])}`")


# =========================================================
# All candidate table
# =========================================================

if show_candidate_table:
    st.subheader("All candidates in this frame")

    # Add ranks for easier diagnosis.
    table_df = candidate_df.copy()
    table_df["score_B_rank"] = table_df["score_B"].rank(ascending=False, method="min").astype(int)
    table_df["sam_score_rank"] = table_df["sam_score"].rank(ascending=False, method="min").astype(int)
    table_df["IoU_rank"] = table_df["IoU_vs_GT"].rank(ascending=False, method="min").astype(int)

    # Optional weighted score column.
    if enable_weighted:
        table_df["weighted_score"] = table_df["idx"].apply(lambda i: float(weighted_score[int(i)]))
        table_df["weighted_rank"] = table_df["weighted_score"].rank(ascending=False, method="min").astype(int)

    preferred_cols = [
        "idx",
        "id",
        "box_id",
        "rank_in_box",
        "sam_score",
        "sam_score_rank",
        "score_B",
        "score_B_rank",
        "A_match",
        "C_match",
        "E_match",
        "S_match",
        "IoU_vs_GT",
        "IoU_rank",
    ]

    if enable_weighted:
        preferred_cols.insert(8, "weighted_score")
        preferred_cols.insert(9, "weighted_rank")

    preferred_cols = [c for c in preferred_cols if c in table_df.columns]

    st.dataframe(
        table_df[preferred_cols],
        use_container_width=True,
        hide_index=True,
    )

    csv = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download this frame's candidate table",
        data=csv,
        file_name=f"{sequence}_{frame_id}_B_candidate_scores.csv",
        mime="text/csv",
    )


# =========================================================
# All candidate image gallery
# =========================================================

if show_candidate_gallery:
    st.subheader("All candidate mask images")

    st.write(
        "Candidates are displayed using the same masks listed in the table above. "
        "The label flags show whether a candidate is the SAM raw-score choice, "
        "the B-score choice, the temporary weighted choice, or the highest-IoU candidate."
    )

    gallery_df = candidate_df.copy()

    if gallery_sort_by not in gallery_df.columns:
        gallery_sort_by = "score_B"

    ascending = gallery_sort_by == "id"
    gallery_df = gallery_df.sort_values(gallery_sort_by, ascending=ascending).reset_index(drop=True)

    max_items = int(min(gallery_max_items, len(gallery_df)))
    gallery_df = gallery_df.head(max_items)

    best_iou_idx = None
    if "IoU_vs_GT" in candidate_df.columns and not candidate_df.empty:
        best_iou_idx = int(candidate_df.loc[candidate_df["IoU_vs_GT"].idxmax(), "idx"])

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
                tags.append("B")
            if enable_weighted and idx == weighted_idx:
                tags.append("W")
            if best_iou_idx is not None and idx == best_iou_idx:
                tags.append("Best IoU")

            tag_text = f" [{' | '.join(tags)}]" if tags else ""

            if show_overlay:
                # Use the same color as the main display for the two important choices.
                # Other candidates use yellow so they are visually distinct from GT/SAM/B.
                if idx == b_idx:
                    color = (0, 0, 255)
                elif idx == sam_idx:
                    color = (255, 0, 0)
                elif best_iou_idx is not None and idx == best_iou_idx:
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
