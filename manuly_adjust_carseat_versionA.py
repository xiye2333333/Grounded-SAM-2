import json
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
st.set_page_config(page_title="ACES-A CarSeat Tuner", layout="wide")
st.title("ACES-A Interactive Tuning UI")

EPS = 1e-8


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
st.sidebar.header("A-version scoring parameters")

# A-version: tune weights and the two term parameters used by the original score.
w_area = st.sidebar.slider(
    "W_AREA",
    min_value=0.000,
    max_value=1.000,
    value=0.150,
    step=0.005,
)
w_center = st.sidebar.slider(
    "W_CENTER",
    min_value=0.000,
    max_value=1.000,
    value=0.200,
    step=0.005,
)
w_edge = st.sidebar.slider(
    "W_EDGE / W_BORDER",
    min_value=0.000,
    max_value=1.000,
    value=0.400,
    step=0.005,
)
w_sil = st.sidebar.slider(
    "W_SILHOUETTE",
    min_value=0.000,
    max_value=1.000,
    value=0.250,
    step=0.005,
)

t_area = st.sidebar.slider(
    "A target: expected area ratio $t_A$",
    min_value=0.005,
    max_value=0.950,
    value=0.250,
    step=0.005,
)
q_border = st.sidebar.slider(
    "E quantile: $q_{border}$",
    min_value=0.000,
    max_value=1.000,
    value=0.300,
    step=0.005,
)

st.sidebar.caption(
    "A score = normalized weighted sum of A/C/E/S. "
    "Defaults: w=(0.15, 0.20, 0.40, 0.25), t_A=0.25, q=0.30."
)

st.sidebar.markdown("---")
show_score_table = st.sidebar.checkbox("Show selected-image score table", value=True)


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
# A-version feature / score functions
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


def normalize_weights(w_raw: np.ndarray) -> np.ndarray:
    w = np.asarray(w_raw, dtype=float)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= EPS:
        return np.asarray([0.25, 0.25, 0.25, 0.25], dtype=float)
    return w / s


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


def compute_a_components(
    mask_bool: np.ndarray,
    W: int,
    H: int,
    dist_map: np.ndarray,
    d_max: float,
    q_border_: float,
    t_area_: float,
    weights_norm: np.ndarray,
) -> Dict[str, float]:
    img_area = W * H
    cx, cy = W / 2.0, H / 2.0

    area_px = int(mask_bool.sum())
    if area_px <= 0:
        return {
            "A_raw": 0.0,
            "A": 0.0,
            "C": 0.0,
            "E": 0.0,
            "S": 0.0,
            "score_A": 0.0,
            "wA": float(weights_norm[0]),
            "wC": float(weights_norm[1]),
            "wE": float(weights_norm[2]),
            "wS": float(weights_norm[3]),
        }

    # A: expected area ratio, using the original parabola centered at t_A.
    A_raw = float(area_px / img_area)
    A = area_term_parabola(A_raw, t_area_)

    # C: 1 means mask centroid is at the image center.
    ys, xs = np.where(mask_bool)
    mx, my = xs.mean(), ys.mean()
    Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
    C = 1.0 - float(np.clip(Dp, 0.0, 1.0))

    # E: original A-version edge/border term. 1 means far from image border.
    q = float(np.quantile(dist_map[mask_bool], q_border_))
    E = float(np.clip(q / d_max, 0.0, 1.0))

    # S: original silhouette quality term.
    S = compute_silhouette_score_v2(mask_bool)

    features = np.asarray([A, C, E, S], dtype=float)
    score_A = float(features @ weights_norm)

    return {
        "A_raw": float(A_raw),
        "A": float(A),
        "C": float(C),
        "E": float(E),
        "S": float(S),
        "score_A": float(np.clip(score_A, 0.0, 1.0)),
        "wA": float(weights_norm[0]),
        "wC": float(weights_norm[1]),
        "wE": float(weights_norm[2]),
        "wS": float(weights_norm[3]),
    }


def compute_a_scores(
    masks_bool: List[np.ndarray],
    W: int,
    H: int,
    q_border_: float,
    t_area_: float,
    weights_raw: np.ndarray,
) -> Tuple[np.ndarray, List[Dict[str, float]], np.ndarray]:
    dist_map = compute_distance_transform(H, W)
    d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0
    weights_norm = normalize_weights(weights_raw)

    components = [
        compute_a_components(
            m,
            W,
            H,
            dist_map,
            d_max,
            q_border_,
            t_area_,
            weights_norm,
        )
        for m in masks_bool
    ]
    scores = np.asarray([c["score_A"] for c in components], dtype=float)
    return scores, components, weights_norm


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

    weights_raw = np.asarray([w_area, w_center, w_edge, w_sil], dtype=float)
    a_scores, components, weights_norm = compute_a_scores(
        masks,
        W,
        H,
        q_border_=q_border,
        t_area_=t_area,
        weights_raw=weights_raw,
    )
    a_idx = int(np.argmax(a_scores))

    gt_path = resolve_gt_path(data, json_fp, gt_root)
    gt_mask = read_gt_cached(str(gt_path)) if gt_path is not None else None

    sam_iou = compute_iou(masks[sam_idx], gt_mask)
    a_iou = compute_iou(masks[a_idx], gt_mask)

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
        "a_scores": a_scores,
        "a_components": components,
        "a_weights_norm": weights_norm,
        "a_idx": a_idx,
        "a_iou": a_iou,
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
with st.spinner("Evaluating current A-version parameters..."):
    for fp in json_files:
        rec = evaluate_one_file(fp, gt_root)
        if rec is not None:
            records.append(rec)

if not records:
    st.warning("Result JSON files were found, but no valid annotations could be decoded.")
    st.stop()

n_total = len(records)
n_with_gt = sum(1 for r in records if r["gt_mask"] is not None and r["sam_iou"] is not None and r["a_iou"] is not None)
n_without_gt = n_total - n_with_gt

sam_mean = mean_or_none([r["sam_iou"] for r in records])
a_mean = mean_or_none([r["a_iou"] for r in records])
delta_mean = None if sam_mean is None or a_mean is None else a_mean - sam_mean

st.markdown("### Performance Overview")
metric_cols = st.columns(5)
metric_cols[0].metric("Images loaded", f"{n_total}")
metric_cols[1].metric("Images with GT", f"{n_with_gt}")
metric_cols[2].metric("SAM mean IoU", "N/A" if sam_mean is None else f"{sam_mean:.4f}")
metric_cols[3].metric("A-score mean IoU", "N/A" if a_mean is None else f"{a_mean:.4f}")
metric_cols[4].metric("Mean Δ", "N/A" if delta_mean is None else f"{delta_mean:+.4f}")

if n_without_gt > 0:
    st.caption(f"{n_without_gt} image(s) have no usable GT or shape-matched GT, so they are excluded from mean IoU.")

weights_norm_current = normalize_weights(np.asarray([w_area, w_center, w_edge, w_sil], dtype=float))
st.caption(
    "Current normalized weights: "
    f"wA={weights_norm_current[0]:.3f}, "
    f"wC={weights_norm_current[1]:.3f}, "
    f"wE={weights_norm_current[2]:.3f}, "
    f"wS={weights_norm_current[3]:.3f}"
)

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
sel_cols = st.columns(4)
sel_cols[0].metric("SAM IoU", fmt_iou(rec["sam_iou"]))
sel_cols[1].metric("A-score IoU", fmt_iou(rec["a_iou"]))
sel_cols[2].metric("SAM best mask", f"{get_mask_id(rec['anns'][rec['sam_idx']], rec['sam_idx'])}")
sel_cols[3].metric("A best mask", f"{get_mask_id(rec['anns'][rec['a_idx']], rec['a_idx'])}")

image = read_image_cached(str(rec["image_path"])) if rec["image_path"] is not None else None
if image is None:
    image = Image.new("RGB", (rec["W"], rec["H"]), (80, 80, 80))

fallback_size = (image.width, image.height)

gt_img = mask_to_pil(rec["gt_mask"], fallback_size=fallback_size)
sam_img = mask_to_pil(rec["masks"][rec["sam_idx"]], fallback_size=fallback_size)
a_img = mask_to_pil(rec["masks"][rec["a_idx"]], fallback_size=fallback_size)

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
    a_img,
    caption=f"A-score best | IoU={fmt_iou(rec['a_iou'])} | A score={rec['a_scores'][rec['a_idx']]:.4f}",
    width="stretch",
)

if show_score_table:
    st.markdown("### Candidate Masks Ranked by A Score")
    rows = []
    order = np.argsort(-rec["a_scores"])
    for rank, idx in enumerate(order, start=1):
        comp = rec["a_components"][int(idx)]
        rows.append({
            "rank_A": rank,
            "mask_id": get_mask_id(rec["anns"][int(idx)], int(idx)),
            "is_SAM_best": int(idx) == rec["sam_idx"],
            "is_A_best": int(idx) == rec["a_idx"],
            "IoU": compute_iou(rec["masks"][int(idx)], rec["gt_mask"]),
            "score_A": comp["score_A"],
            "A_raw": comp["A_raw"],
            "A": comp["A"],
            "C": comp["C"],
            "E": comp["E"],
            "S": comp["S"],
            "wA_norm": comp["wA"],
            "wC_norm": comp["wC"],
            "wE_norm": comp["wE"],
            "wS_norm": comp["wS"],
            "SAM_score": rec["sam_scores"][int(idx)],
        })
    st.dataframe(rows, width="stretch", hide_index=True)

st.markdown("---")
st.caption(
    "A score = wA*A + wC*C + wE*E + wS*S after nonnegative weight normalization. "
    "A uses the area parabola centered at t_A. C is center proximity. "
    "E is the original distance-from-border term, so larger E means farther from the image edge. "
    "S is silhouette coherence."
)
