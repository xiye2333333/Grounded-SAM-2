import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pycocotools.mask as mask_util
import streamlit as st
from PIL import Image


# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="ACES-B CarSeat Tuner", layout="wide")
st.title("ACES-B Interactive Tuning UI with Candidate + GT Score/Term Means")

EPS = 1e-8
Q_BORDER_FIXED = 0.25


# =========================================================
# Dataset / UI settings
# =========================================================
DEFAULT_GT_DIR = r"D:\uwb thesis\code\Grounded-SAM-2\GT_masks"

AVAILABLE_DATASETS = {
    "CarSeat / custom result folder": "",
    "outputs/AdjustWeightData": "outputs/AdjustWeightData",
    "outputs/CatagroiesAnalyze/BackgroundNoise": "outputs/CatagroiesAnalyze/BackgroundNoise",
    "outputs/CatagroiesAnalyze/FrontNoise": "outputs/CatagroiesAnalyze/FrontNoise",
    "outputs/CatagroiesAnalyze/MutiObject": "outputs/CatagroiesAnalyze/MutiObject",
    "outputs/CatagroiesAnalyze/SepratePieces": "outputs/CatagroiesAnalyze/SepratePieces",
    "outputs/CatagroiesAnalyze/ShallowAndHoles": "outputs/CatagroiesAnalyze/ShallowAndHoles",
    "outputs/AllMasks_v2_score_record": "outputs/AllMasks_v2_score_record",
}

st.sidebar.header("Dataset")
dataset_name = st.sidebar.selectbox("Select dataset", list(AVAILABLE_DATASETS.keys()), index=0)

initial_json_dir = AVAILABLE_DATASETS[dataset_name]
json_dir = st.sidebar.text_input(
    "Result JSON folder",
    value=initial_json_dir,
    placeholder=r"D:\...\CarSeat_results",
)

gt_dir = st.sidebar.text_input("GT mask/json folder", value=DEFAULT_GT_DIR)
recursive_search = st.sidebar.checkbox("Search JSON files recursively", value=True)

st.sidebar.markdown("---")
st.sidebar.header("B-version hypothesis parameters")

# Four tunable bars only. q_border is intentionally fixed at 0.25.
t_area = st.sidebar.slider(
    "A target: expected area ratio $t_A$",
    min_value=0.005,
    max_value=0.950,
    value=0.250,
    step=0.005,
)
c_target = st.sidebar.slider(
    "C target: expected center proximity $c$",
    min_value=0.000,
    max_value=1.000,
    value=0.800,
    step=0.005,
)
e_target = st.sidebar.slider(
    "E target: expected edge proximity $e$",
    min_value=0.000,
    max_value=1.000,
    value=0.000,
    step=0.005,
)
s_target = st.sidebar.slider(
    "S target: expected silhouette quality $s$",
    min_value=0.000,
    max_value=1.000,
    value=0.900,
    step=0.005,
)

st.sidebar.caption(f"q_border is fixed at {Q_BORDER_FIXED:.2f} for this B-version test.")

st.sidebar.markdown("---")
show_score_table = st.sidebar.checkbox("Show selected-image score table", value=True)
show_candidate_gallery = st.sidebar.checkbox("Show all candidate mask images at bottom", value=True)
gallery_cols_per_row = st.sidebar.slider("Candidate gallery columns per row", 2, 6, 4, 1)
gallery_sort_by = st.sidebar.selectbox(
    "Candidate gallery sort by",
    options=["B score", "IoU", "SAM score", "mask id"],
    index=0,
)


# =========================================================
# COCO RLE / mask utilities
# =========================================================
def _normalize_rle_for_decode(rle: Dict[str, Any]) -> Dict[str, Any]:
    """pycocotools is safest when compressed RLE counts are bytes."""
    rle2 = dict(rle)
    counts = rle2.get("counts")
    if isinstance(counts, str):
        rle2["counts"] = counts.encode("utf-8")
    return rle2


def decode_rle_mask_bool(rle: Dict[str, Any]) -> np.ndarray:
    m = mask_util.decode(_normalize_rle_for_decode(rle))
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)


def load_gt_mask_from_json(gt_json_path: Path) -> Optional[np.ndarray]:
    try:
        data = json.loads(gt_json_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    # Support both this project format and common annotation-style formats.
    for key in ("segmentation_rle", "segmentation", "rle"):
        rle = data.get(key)
        if isinstance(rle, dict) and "counts" in rle and "size" in rle:
            return decode_rle_mask_bool(rle)

    return None


def load_gt_mask(gt_path: Path) -> Optional[np.ndarray]:
    if not gt_path.exists():
        return None

    if gt_path.suffix.lower() == ".json":
        return load_gt_mask_from_json(gt_path)

    try:
        arr = np.array(Image.open(gt_path))
    except Exception:
        return None

    if arr.ndim == 2:
        return (arr > 0).astype(bool)
    return (arr.sum(axis=2) > 0).astype(bool)


def compute_iou(a: np.ndarray, b: Optional[np.ndarray]) -> Optional[float]:
    if b is None:
        return None
    if a.shape != b.shape:
        return None
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def mask_to_pil(mask: Optional[np.ndarray], fallback_size: Tuple[int, int] = (480, 360)) -> Image.Image:
    if mask is None:
        return Image.new("L", fallback_size, 0)
    return Image.fromarray((mask.astype(np.uint8) * 255))


def get_sam_score(ann: Dict[str, Any], fallback: float = 0.0) -> float:
    for key in ("sam_score", "score_sam", "score"):
        if key in ann:
            try:
                return float(ann[key])
            except Exception:
                pass
    return float(fallback)


def get_mask_id(ann: Dict[str, Any], idx: int) -> int:
    for key in ("id", "mask_index", "index"):
        if key in ann:
            try:
                return int(ann[key])
            except Exception:
                pass
    return int(idx)


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


def compute_b_components(
    mask_bool: np.ndarray,
    W: int,
    H: int,
    dist_map: np.ndarray,
    d_max: float,
    q_border: float,
    t_area_: float,
    c_target_: float,
    e_target_: float,
    s_target_: float,
) -> Dict[str, float]:
    img_area = W * H
    cx, cy = W / 2.0, H / 2.0

    area_px = int(mask_bool.sum())
    if area_px <= 0:
        return {
            "A_raw": 0.0,
            "A_match": 0.0,
            "C_obs": 0.0,
            "C_match": 0.0,
            "E_dist_obs": 0.0,
            "E_edge_obs": 0.0,
            "E_match": 0.0,
            "S_obs": 0.0,
            "S_match": 0.0,
            "score_B": 0.0,
        }

    # A: existing parabola already measures match to expected size t_A.
    A_raw = float(area_px / img_area)
    A_match = area_term_parabola(A_raw, t_area_)

    # C_obs: 1 means at image center, 0 means maximally far from image center.
    ys, xs = np.where(mask_bool)
    mx, my = xs.mean(), ys.mean()
    Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
    C_obs = 1.0 - float(np.clip(Dp, 0.0, 1.0))
    C_match = 1.0 - abs(float(c_target_) - C_obs)

    # Original E_dist_obs: 1 means far from image edge/border.
    # B-version e_target means edge proximity, so reverse it first.
    q = float(np.quantile(dist_map[mask_bool], q_border))
    E_dist_obs = float(np.clip(q / d_max, 0.0, 1.0))
    E_edge_obs = 1.0 - E_dist_obs
    E_match = 1.0 - abs(float(e_target_) - E_edge_obs)

    # S_obs: 1 means clean / closed / non-fragmented silhouette.
    S_obs = compute_silhouette_score_v2(mask_bool)
    S_match = 1.0 - abs(float(s_target_) - S_obs)

    A_match = float(np.clip(A_match, 0.0, 1.0))
    C_match = float(np.clip(C_match, 0.0, 1.0))
    E_match = float(np.clip(E_match, 0.0, 1.0))
    S_match = float(np.clip(S_match, 0.0, 1.0))

    score_B = A_match + C_match + E_match + S_match

    return {
        "A_raw": float(A_raw),
        "A_match": A_match,
        "C_obs": float(C_obs),
        "C_match": C_match,
        "E_dist_obs": float(E_dist_obs),
        "E_edge_obs": float(E_edge_obs),
        "E_match": E_match,
        "S_obs": float(S_obs),
        "S_match": S_match,
        "score_B": float(score_B),
    }


def compute_b_scores(
    masks_bool: List[np.ndarray],
    W: int,
    H: int,
    q_border: float,
    t_area_: float,
    c_target_: float,
    e_target_: float,
    s_target_: float,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    dist_map = compute_distance_transform(H, W)
    d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

    components = [
        compute_b_components(
            m,
            W,
            H,
            dist_map,
            d_max,
            q_border,
            t_area_,
            c_target_,
            e_target_,
            s_target_,
        )
        for m in masks_bool
    ]
    scores = np.asarray([c["score_B"] for c in components], dtype=float)
    return scores, components


# =========================================================
# File loading helpers
# =========================================================
def collect_json_files(folder: Path, recursive: bool) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []

    files = sorted(folder.rglob("*_result.json") if recursive else folder.glob("*_result.json"))
    # Avoid picking backup files accidentally.
    return [p for p in files if not p.name.endswith(".bak")]


@st.cache_data(show_spinner=False)
def read_json_cached(path_str: str) -> Dict[str, Any]:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_image_cached(path_str: str) -> Optional[Image.Image]:
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def read_gt_cached(path_str: str) -> Optional[np.ndarray]:
    return load_gt_mask(Path(path_str))


def resolve_image_path(data: Dict[str, Any], json_fp: Path) -> Optional[Path]:
    candidates: List[Path] = []

    image_path = data.get("image_path")
    if isinstance(image_path, str) and image_path.strip():
        candidates.append(Path(image_path))

        # If an absolute Windows path is not valid from the current working dir,
        # still try the basename next to the JSON file.
        candidates.append(json_fp.parent / Path(image_path).name)

    key = json_fp.stem.replace("_result", "")
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidates.append(json_fp.parent / f"{key}{ext}")
        candidates.append(json_fp.parent.parent / f"{key}{ext}")

    for p in candidates:
        if p.exists():
            return p
    return None


def candidate_gt_paths(data: Dict[str, Any], json_fp: Path, gt_root: Path) -> List[Path]:
    image_path = data.get("image_path")
    if isinstance(image_path, str) and image_path.strip():
        image_stem = Path(image_path).stem
    else:
        image_stem = json_fp.stem.replace("_result", "")

    json_stem = json_fp.stem.replace("_result", "")
    stems = []
    for s in (image_stem, json_stem):
        if s and s not in stems:
            stems.append(s)

    candidates: List[Path] = []
    for stem in stems:
        # User-provided convention: img_xxxx_gt.png / img_xxxx_gt.json
        for ext in (".png", ".jpg", ".jpeg", ".json"):
            candidates.append(gt_root / f"{stem}_gt{ext}")
        # Fallback in case the stem already includes _gt or the GT is named directly.
        for ext in (".png", ".jpg", ".jpeg", ".json"):
            candidates.append(gt_root / f"{stem}{ext}")

    return candidates


def resolve_gt_path(data: Dict[str, Any], json_fp: Path, gt_root: Path) -> Optional[Path]:
    for p in candidate_gt_paths(data, json_fp, gt_root):
        if p.exists():
            return p
    return None


def decode_annotations(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    anns = data.get("annotations", [])
    valid_anns: List[Dict[str, Any]] = []
    masks: List[np.ndarray] = []

    for ann in anns:
        rle = ann.get("segmentation")
        if not isinstance(rle, dict):
            continue
        try:
            m = decode_rle_mask_bool(rle)
        except Exception:
            continue
        valid_anns.append(ann)
        masks.append(m)

    return valid_anns, masks


def evaluate_one_file(json_fp: Path, gt_root: Path) -> Optional[Dict[str, Any]]:
    try:
        data = read_json_cached(str(json_fp))
    except Exception:
        return None

    anns, masks = decode_annotations(data)
    if not anns:
        return None

    H = int(data.get("img_height", masks[0].shape[0]))
    W = int(data.get("img_width", masks[0].shape[1]))

    # Use actual decoded mask shape if the metadata is absent or inconsistent.
    if masks[0].shape != (H, W):
        H, W = masks[0].shape

    sam_scores = np.asarray([get_sam_score(a) for a in anns], dtype=float)
    sam_idx = int(np.argmax(sam_scores))

    b_scores, components = compute_b_scores(
        masks,
        W,
        H,
        q_border=Q_BORDER_FIXED,
        t_area_=t_area,
        c_target_=c_target,
        e_target_=e_target,
        s_target_=s_target,
    )
    b_idx = int(np.argmax(b_scores))

    gt_path = resolve_gt_path(data, json_fp, gt_root)
    gt_mask = read_gt_cached(str(gt_path)) if gt_path is not None else None

    gt_components = None
    gt_score = None
    if gt_mask is not None and gt_mask.shape == (H, W):
        gt_scores, gt_component_list = compute_b_scores(
            [gt_mask],
            W,
            H,
            q_border=Q_BORDER_FIXED,
            t_area_=t_area,
            c_target_=c_target,
            e_target_=e_target,
            s_target_=s_target,
        )
        gt_score = float(gt_scores[0])
        gt_components = gt_component_list[0]

    sam_iou = compute_iou(masks[sam_idx], gt_mask)
    b_iou = compute_iou(masks[b_idx], gt_mask)

    image_path = resolve_image_path(data, json_fp)

    return {
        "json_fp": json_fp,
        "data": data,
        "anns": anns,
        "masks": masks,
        "H": H,
        "W": W,
        "image_path": image_path,
        "gt_path": gt_path,
        "gt_mask": gt_mask,
        "sam_scores": sam_scores,
        "sam_idx": sam_idx,
        "sam_iou": sam_iou,
        "b_scores": b_scores,
        "b_components": components,
        "b_idx": b_idx,
        "b_iou": b_iou,
        "gt_score": gt_score,
        "gt_components": gt_components,
    }


def fmt_iou(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{x:.4f}"


def mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    return float(arr.mean()) if arr.size else None


# =========================================================
# Main app logic
# =========================================================
if not json_dir.strip():
    st.info("Select or enter a result JSON folder in the left sidebar.")
    st.stop()

json_root = Path(json_dir)
gt_root = Path(gt_dir)
json_files = collect_json_files(json_root, recursive_search)

if not json_files:
    st.warning("No *_result.json files found in the selected result folder.")
    st.stop()

records: List[Dict[str, Any]] = []
with st.spinner("Evaluating current B-version parameters..."):
    for fp in json_files:
        rec = evaluate_one_file(fp, gt_root)
        if rec is not None:
            records.append(rec)

if not records:
    st.warning("Result JSON files were found, but no valid annotations could be decoded.")
    st.stop()

n_total = len(records)
n_with_gt = sum(1 for r in records if r["gt_mask"] is not None and r["sam_iou"] is not None and r["b_iou"] is not None)
n_without_gt = n_total - n_with_gt

sam_mean = mean_or_none([r["sam_iou"] for r in records])
b_mean = mean_or_none([r["b_iou"] for r in records])
delta_mean = None if sam_mean is None or b_mean is None else b_mean - sam_mean

st.markdown("### Performance Overview")
metric_cols = st.columns(5)
metric_cols[0].metric("Images loaded", f"{n_total}")
metric_cols[1].metric("Images with GT", f"{n_with_gt}")
metric_cols[2].metric("SAM mean IoU", "N/A" if sam_mean is None else f"{sam_mean:.4f}")
metric_cols[3].metric("B-score mean IoU", "N/A" if b_mean is None else f"{b_mean:.4f}")
metric_cols[4].metric("Mean Δ", "N/A" if delta_mean is None else f"{delta_mean:+.4f}")

if n_without_gt > 0:
    st.caption(f"{n_without_gt} image(s) have no usable GT or shape-matched GT, so they are excluded from mean IoU.")

# ---------------------------------------------------------
# B score / term-value overview across all candidate masks
# ---------------------------------------------------------
all_components = []
for r in records:
    all_components.extend(r["b_components"])

gt_components_all = [r["gt_components"] for r in records if r.get("gt_components") is not None]

def component_mean(components: List[Dict[str, float]], key: str) -> Optional[float]:
    vals = [float(c[key]) for c in components if c is not None and key in c]
    return float(np.mean(vals)) if vals else None

def component_std(components: List[Dict[str, float]], key: str) -> Optional[float]:
    vals = [float(c[key]) for c in components if c is not None and key in c]
    return float(np.std(vals)) if vals else None

def comp_mean(key: str) -> Optional[float]:
    return component_mean(all_components, key)

def comp_std(key: str) -> Optional[float]:
    return component_std(all_components, key)

def gt_comp_mean(key: str) -> Optional[float]:
    return component_mean(gt_components_all, key)

def gt_comp_std(key: str) -> Optional[float]:
    return component_std(gt_components_all, key)

def fmt_metric(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{x:.4f}"

st.markdown("### B Score / Term Means Across All Candidate Masks")
st.caption(
    "These averages are computed over every decoded candidate mask, not only over the B-selected masks "
    "and not only over images with GT. This is useful for checking the actual score scale under the current Parameter-B setting."
)
score_cols = st.columns(5)
score_cols[0].metric("Mean score_B", fmt_metric(comp_mean("score_B")))
score_cols[1].metric("Mean A_match", fmt_metric(comp_mean("A_match")))
score_cols[2].metric("Mean C_match", fmt_metric(comp_mean("C_match")))
score_cols[3].metric("Mean E_match", fmt_metric(comp_mean("E_match")))
score_cols[4].metric("Mean S_match", fmt_metric(comp_mean("S_match")))

with st.expander("Show detailed B score / term statistics", expanded=False):
    detail_rows = [
        {
            "quantity": "score_B",
            "meaning": "final B score = A_match + C_match + E_match + S_match",
            "mean": comp_mean("score_B"),
            "std": comp_std("score_B"),
        },
        {
            "quantity": "A_raw",
            "meaning": "observed mask area ratio",
            "mean": comp_mean("A_raw"),
            "std": comp_std("A_raw"),
        },
        {
            "quantity": "A_match",
            "meaning": "area match to t_area",
            "mean": comp_mean("A_match"),
            "std": comp_std("A_match"),
        },
        {
            "quantity": "C_obs",
            "meaning": "observed center proximity; 1 = image center",
            "mean": comp_mean("C_obs"),
            "std": comp_std("C_obs"),
        },
        {
            "quantity": "C_match",
            "meaning": "center match to c_target",
            "mean": comp_mean("C_match"),
            "std": comp_std("C_match"),
        },
        {
            "quantity": "E_edge_obs",
            "meaning": "observed edge proximity; 1 = close to image edge",
            "mean": comp_mean("E_edge_obs"),
            "std": comp_std("E_edge_obs"),
        },
        {
            "quantity": "E_match",
            "meaning": "edge-proximity match to e_target",
            "mean": comp_mean("E_match"),
            "std": comp_std("E_match"),
        },
        {
            "quantity": "S_obs",
            "meaning": "observed silhouette quality",
            "mean": comp_mean("S_obs"),
            "std": comp_std("S_obs"),
        },
        {
            "quantity": "S_match",
            "meaning": "silhouette match to s_target",
            "mean": comp_mean("S_match"),
            "std": comp_std("S_match"),
        },
    ]
    st.dataframe(detail_rows, width="stretch", hide_index=True)

# ---------------------------------------------------------
# B score / term-value overview for GT masks only
# ---------------------------------------------------------
st.markdown("### B Score / Term Means Across GT Masks")
st.caption(
    "These values treat each available GT mask as an input mask and compute the same B-version score/terms. "
    "This shows whether the real target masks themselves match the current Parameter-B assumptions. "
    "Unlike IoU, this is not a correctness metric; it is a hypothesis-fit score."
)
gt_cols = st.columns(6)
gt_cols[0].metric("GT masks counted", f"{len(gt_components_all)}")
gt_cols[1].metric("Mean GT score_B", fmt_metric(gt_comp_mean("score_B")))
gt_cols[2].metric("Mean GT A_match", fmt_metric(gt_comp_mean("A_match")))
gt_cols[3].metric("Mean GT C_match", fmt_metric(gt_comp_mean("C_match")))
gt_cols[4].metric("Mean GT E_match", fmt_metric(gt_comp_mean("E_match")))
gt_cols[5].metric("Mean GT S_match", fmt_metric(gt_comp_mean("S_match")))

with st.expander("Show detailed GT B score / term statistics", expanded=False):
    gt_detail_rows = [
        {
            "quantity": "score_B",
            "meaning": "final GT B score = A_match + C_match + E_match + S_match",
            "mean": gt_comp_mean("score_B"),
            "std": gt_comp_std("score_B"),
        },
        {
            "quantity": "A_raw",
            "meaning": "GT area ratio",
            "mean": gt_comp_mean("A_raw"),
            "std": gt_comp_std("A_raw"),
        },
        {
            "quantity": "A_match",
            "meaning": "GT area match to t_area",
            "mean": gt_comp_mean("A_match"),
            "std": gt_comp_std("A_match"),
        },
        {
            "quantity": "C_obs",
            "meaning": "GT center proximity; 1 = image center",
            "mean": gt_comp_mean("C_obs"),
            "std": gt_comp_std("C_obs"),
        },
        {
            "quantity": "C_match",
            "meaning": "GT center match to c_target",
            "mean": gt_comp_mean("C_match"),
            "std": gt_comp_std("C_match"),
        },
        {
            "quantity": "E_edge_obs",
            "meaning": "GT edge proximity; 1 = close to image edge",
            "mean": gt_comp_mean("E_edge_obs"),
            "std": gt_comp_std("E_edge_obs"),
        },
        {
            "quantity": "E_match",
            "meaning": "GT edge-proximity match to e_target",
            "mean": gt_comp_mean("E_match"),
            "std": gt_comp_std("E_match"),
        },
        {
            "quantity": "S_obs",
            "meaning": "GT silhouette quality",
            "mean": gt_comp_mean("S_obs"),
            "std": gt_comp_std("S_obs"),
        },
        {
            "quantity": "S_match",
            "meaning": "GT silhouette match to s_target",
            "mean": gt_comp_mean("S_match"),
            "std": gt_comp_std("S_match"),
        },
    ]
    st.dataframe(gt_detail_rows, width="stretch", hide_index=True)

st.markdown("---")

# Select one image for detailed display.
labels = []
for i, r in enumerate(records):
    gt_tag = "GT" if r["gt_mask"] is not None and r["sam_iou"] is not None else "no GT"
    labels.append(f"{i:04d} | {r['json_fp'].name} | {gt_tag}")

selected_label = st.selectbox("Select image", labels, index=0)
selected_idx = labels.index(selected_label)
rec = records[selected_idx]

st.markdown("### Selected Image")
sel_cols = st.columns(5)
sel_cols[0].metric("SAM IoU", fmt_iou(rec["sam_iou"]))
sel_cols[1].metric("B-score IoU", fmt_iou(rec["b_iou"]))
sel_cols[2].metric("GT B score", fmt_metric(rec.get("gt_score")))
sel_cols[3].metric("SAM best mask", f"{get_mask_id(rec['anns'][rec['sam_idx']], rec['sam_idx'])}")
sel_cols[4].metric("B best mask", f"{get_mask_id(rec['anns'][rec['b_idx']], rec['b_idx'])}")

image = read_image_cached(str(rec["image_path"])) if rec["image_path"] is not None else None
if image is None:
    image = Image.new("RGB", (rec["W"], rec["H"]), (80, 80, 80))

fallback_size = (image.width, image.height)

gt_img = mask_to_pil(rec["gt_mask"], fallback_size=fallback_size)
sam_img = mask_to_pil(rec["masks"][rec["sam_idx"]], fallback_size=fallback_size)
b_img = mask_to_pil(rec["masks"][rec["b_idx"]], fallback_size=fallback_size)

view_cols = st.columns(4)
view_cols[0].image(image, caption=f"Original: {rec['image_path'] if rec['image_path'] else 'path not found'}", width="stretch")
if rec["gt_mask"] is None:
    view_cols[1].image(gt_img, caption="GT: not found", width="stretch")
else:
    view_cols[1].image(gt_img, caption=f"GT: {rec['gt_path']}", width="stretch")
view_cols[2].image(
    sam_img,
    caption=f"SAM best | IoU={fmt_iou(rec['sam_iou'])} | SAM score={rec['sam_scores'][rec['sam_idx']]:.4f}",
    width="stretch",
)
view_cols[3].image(
    b_img,
    caption=f"B-score best | IoU={fmt_iou(rec['b_iou'])} | B score={rec['b_scores'][rec['b_idx']]:.4f}",
    width="stretch",
)

if rec.get("gt_components") is not None:
    st.markdown("### Selected Image GT B Score / Terms")
    gt_comp = rec["gt_components"]
    gt_row = [{
        "mask_type": "GT",
        "score_B": gt_comp["score_B"],
        "A_raw": gt_comp["A_raw"],
        "A_match": gt_comp["A_match"],
        "C_obs": gt_comp["C_obs"],
        "C_match": gt_comp["C_match"],
        "E_edge_obs": gt_comp["E_edge_obs"],
        "E_match": gt_comp["E_match"],
        "S_obs": gt_comp["S_obs"],
        "S_match": gt_comp["S_match"],
    }]
    st.dataframe(gt_row, width="stretch", hide_index=True)
else:
    st.caption("No shape-matched GT mask is available for computing GT B score/terms for this image.")

def build_candidate_rows_for_selected_image(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    order = np.argsort(-rec["b_scores"])
    for rank, idx in enumerate(order, start=1):
        idx = int(idx)
        comp = rec["b_components"][idx]
        rows.append({
            "rank_B": rank,
            "candidate_index": idx,
            "mask_id": get_mask_id(rec["anns"][idx], idx),
            "is_SAM_best": idx == rec["sam_idx"],
            "is_B_best": idx == rec["b_idx"],
            "IoU": compute_iou(rec["masks"][idx], rec["gt_mask"]),
            "score_B": comp["score_B"],
            "A_raw": comp["A_raw"],
            "A_match": comp["A_match"],
            "C_obs": comp["C_obs"],
            "C_match": comp["C_match"],
            "E_edge_obs": comp["E_edge_obs"],
            "E_match": comp["E_match"],
            "S_obs": comp["S_obs"],
            "S_match": comp["S_match"],
            "SAM_score": rec["sam_scores"][idx],
        })
    return rows


candidate_rows = build_candidate_rows_for_selected_image(rec)

if show_score_table:
    st.markdown("### Candidate Masks Ranked by B Score")
    st.dataframe(candidate_rows, width="stretch", hide_index=True)

if show_candidate_gallery:
    st.markdown("### All Candidate Mask Images")
    st.caption(
        "Every decoded candidate mask for the selected image is shown below. "
        "Tags indicate the SAM raw-score choice, the current B-score choice, and the best-IoU candidate when GT is available."
    )

    gallery_rows = list(candidate_rows)

    def _neg_inf_if_none(x):
        return -np.inf if x is None else float(x)

    if gallery_sort_by == "B score":
        gallery_rows = sorted(gallery_rows, key=lambda r: _neg_inf_if_none(r["score_B"]), reverse=True)
    elif gallery_sort_by == "IoU":
        gallery_rows = sorted(gallery_rows, key=lambda r: _neg_inf_if_none(r["IoU"]), reverse=True)
    elif gallery_sort_by == "SAM score":
        gallery_rows = sorted(gallery_rows, key=lambda r: _neg_inf_if_none(r["SAM_score"]), reverse=True)
    elif gallery_sort_by == "mask id":
        gallery_rows = sorted(gallery_rows, key=lambda r: int(r["mask_id"]))

    best_iou_idx = None
    iou_values = [r["IoU"] for r in candidate_rows]
    if any(v is not None for v in iou_values):
        valid_iou_rows = [r for r in candidate_rows if r["IoU"] is not None]
        if valid_iou_rows:
            best_iou_idx = int(max(valid_iou_rows, key=lambda r: float(r["IoU"]))["candidate_index"])

    cols_per_row = int(gallery_cols_per_row)
    for start in range(0, len(gallery_rows), cols_per_row):
        cols = st.columns(cols_per_row)
        chunk = gallery_rows[start:start + cols_per_row]

        for col, row in zip(cols, chunk):
            idx = int(row["candidate_index"])
            mask_img = mask_to_pil(rec["masks"][idx], fallback_size=fallback_size)

            tags = []
            if idx == rec["sam_idx"]:
                tags.append("SAM")
            if idx == rec["b_idx"]:
                tags.append("B")
            if best_iou_idx is not None and idx == best_iou_idx:
                tags.append("Best IoU")
            tag_text = f" [{' | '.join(tags)}]" if tags else ""

            with col:
                st.markdown(f"**rank {int(row['rank_B'])} | mask {int(row['mask_id'])}{tag_text}**")
                st.image(mask_img, width="stretch")
                st.caption(
                    f"idx={idx} | IoU={fmt_iou(row['IoU'])} | "
                    f"B={row['score_B']:.4f} | SAM={row['SAM_score']:.4f}"
                )
                st.caption(
                    f"A={row['A_match']:.4f}, C={row['C_match']:.4f}, "
                    f"E={row['E_match']:.4f}, S={row['S_match']:.4f}"
                )

st.markdown("---")
st.caption(
    "B score = A_match + C_match + E_match + S_match. "
    "A_match uses the original area parabola. C/E/S use 1 - abs(target - observed). "
    "E_target is edge proximity, so the original distance-from-border E value is reversed before matching. GT score/terms apply the same formula to the GT mask when available."
)
