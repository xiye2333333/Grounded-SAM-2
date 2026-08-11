import os
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import cv2
import streamlit as st
from PIL import Image
import pycocotools.mask as mask_util

# =========================
# Streamlit UI config
# =========================
st.set_page_config(layout="wide")
st.title("DAVIS Score Tuner (SAM2+DINO candidates)")

EPS = 1e-8


# =========================
# RLE utils
# =========================
def rle_to_mask_bool(rle_obj: Dict[str, Any]) -> np.ndarray:
    """rle_obj: {"size":[H,W], "counts": str} or counts bytes"""
    counts = rle_obj["counts"]
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    rle = {"size": rle_obj["size"], "counts": counts}
    m = mask_util.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)

def mask_to_u8(mask_bool: np.ndarray) -> np.ndarray:
    return (mask_bool.astype(np.uint8) * 255)

def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


# =========================
# Score features
# =========================
def compute_distance_transform(h: int, w: int) -> np.ndarray:
    border_mask = np.zeros((h, w), np.uint8)
    border_mask[1:-1, 1:-1] = 1
    return cv2.distanceTransform(border_mask, distanceType=cv2.DIST_L2, maskSize=5)

def area_term_parabola(x: float, d: float, eps: float = 1e-6) -> float:
    d = float(np.clip(d, eps, 1.0))
    x = float(np.clip(x, 0.0, 1.0))
    val = 1.0 - ((x - d) / d) ** 2
    return float(np.clip(val, 0.0, 1.0))

def compute_silhouette_score_v2(mask_bool: np.ndarray) -> float:
    mask_u8 = (mask_bool.astype(np.uint8) > 0).astype(np.uint8)

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
    h2, w2 = bg_pad.shape
    ff_mask = np.zeros((h2 + 2, w2 + 2), dtype=np.uint8)
    bg_ff = bg_pad.copy()
    cv2.floodFill(bg_ff, ff_mask, (0, 0), 0)

    holes_map = (bg_ff[1:-1, 1:-1] == 1)
    holes_area = float(np.count_nonzero(holes_map))
    hole_ratio = float(np.clip(holes_area / area, 0.0, 1.0)) if area > 0 else 0.0

    sil = solidity * (1.0 - fragmentation) * (1.0 - hole_ratio)
    return float(np.clip(sil, 0.0, 1.0))

def normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= EPS:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    return w / s

def mask_features(mask_bool: np.ndarray, W: int, H: int, dist_map: np.ndarray, d_max: float,
                  q_border: float, t_area: float) -> np.ndarray:
    img_area = W * H
    cx, cy = W / 2, H / 2

    area_px = int(mask_bool.sum())
    if area_px <= 0:
        return np.array([0, 0, 0, 0], dtype=float)

    # A: area preference around t_area
    A_raw = area_px / img_area
    A = area_term_parabola(A_raw, t_area)

    # C: center proximity (1 - normalized distance)
    ys, xs = np.where(mask_bool)
    mx, my = xs.mean(), ys.mean()
    Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
    C = 1.0 - float(np.clip(Dp, 0.0, 1.0))

    # E: border distance quantile
    q = float(np.quantile(dist_map[mask_bool], q_border))
    E = float(np.clip(q / d_max, 0.0, 1.0))

    # Sil
    Sil = compute_silhouette_score_v2(mask_bool)

    return np.array([A, C, E, Sil], dtype=float)

def overlay_mask_on_rgb(rgb: np.ndarray, mask_bool: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """rgb uint8 HxWx3"""
    out = rgb.copy()
    color = np.array([0, 255, 0], dtype=np.uint8)  # green
    m = mask_bool.astype(bool)
    out[m] = (out[m] * (1 - alpha) + color * alpha).astype(np.uint8)
    return out


# =========================
# Data loading
# =========================
@st.cache_data(show_spinner=False)
def find_davis_root(davis_root: str) -> str:
    req = ["JPEGImages", "Annotations", "ImageSets"]
    for r in [davis_root, os.path.join(davis_root, "DAVIS")]:
        if all(os.path.isdir(os.path.join(r, k)) for k in req):
            return r
    raise FileNotFoundError(f"Bad DAVIS root: {davis_root}")

@st.cache_data(show_spinner=False)
def list_sequences(davis_root: str, resolution: str) -> List[str]:
    p = Path(davis_root) / "JPEGImages" / resolution
    if not p.exists():
        return []
    seqs = sorted([d.name for d in p.iterdir() if d.is_dir()])
    return seqs

@st.cache_data(show_spinner=False)
def list_frames(davis_root: str, resolution: str, seq: str) -> List[str]:
    p = Path(davis_root) / "JPEGImages" / resolution / seq
    frames = sorted([f.name for f in p.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    return frames

@st.cache_data(show_spinner=False)
def load_result_json(result_fp: str) -> Dict[str, Any]:
    return json.loads(Path(result_fp).read_text(encoding="utf-8"))

@st.cache_data(show_spinner=False)
def load_gt_from_gtjson(gt_json_fp: str) -> Optional[np.ndarray]:
    p = Path(gt_json_fp)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    rle = data.get("segmentation_rle", None)
    if isinstance(rle, dict):
        return rle_to_mask_bool(rle)
    return None

@st.cache_data(show_spinner=False)
def load_gt_from_davis_png(gt_png_fp: str) -> Optional[np.ndarray]:
    p = Path(gt_png_fp)
    if not p.exists():
        return None
    m = np.array(Image.open(p))
    if m.ndim == 2:
        fg = (m > 0)
    else:
        fg = (m.sum(axis=2) > 0)
    return fg.astype(bool)

@st.cache_data(show_spinner=False)
def load_rgb_image(img_fp: str) -> np.ndarray:
    img = Image.open(img_fp).convert("RGB")
    return np.array(img, dtype=np.uint8)

@st.cache_data(show_spinner=False)
def get_dist_map(H: int, W: int) -> Tuple[np.ndarray, float]:
    dist = compute_distance_transform(H, W)
    dmax = float(dist.max()) if dist.max() > 0 else 1.0
    return dist, dmax

@st.cache_data(show_spinner=False)
def precompute_features_for_frame(result_fp: str, q_border: float, t_area: float) -> Dict[str, Any]:
    """
    Cache-friendly: this depends on (result_fp, q_border, t_area) because A/E features depend on them.
    Returns:
      - X: (K,4) features
      - sam_scores: (K,)
      - rles: list of RLE dicts
      - ids: list of int ids
      - H,W
    """
    data = load_result_json(result_fp)
    anns = data.get("annotations", [])
    H = int(data.get("img_height", 0))
    W = int(data.get("img_width", 0))
    if H <= 0 or W <= 0:
        # fallback: try rle size
        if anns and "segmentation" in anns[0]:
            H, W = anns[0]["segmentation"]["size"]
            H, W = int(H), int(W)

    dist_map, dmax = get_dist_map(H, W)

    rles = []
    ids = []
    sam_scores = []
    feats = []

    for ann in anns:
        rle = ann.get("segmentation", None)
        if not isinstance(rle, dict):
            continue
        m = rle_to_mask_bool(rle)
        f = mask_features(m, W, H, dist_map, dmax, q_border=q_border, t_area=t_area)

        rles.append(rle)
        ids.append(int(ann.get("id", ann.get("mask_index", len(ids)))))
        sam_scores.append(float(ann.get("sam_score", ann.get("score", 0.0))))
        feats.append(f)

    if len(feats) == 0:
        X = np.zeros((0, 4), dtype=float)
    else:
        X = np.stack(feats, axis=0).astype(float)

    return {
        "X": X,
        "sam_scores": np.asarray(sam_scores, dtype=float),
        "rles": rles,
        "ids": ids,
        "H": H,
        "W": W,
        "text_prompt": data.get("text_prompt", ""),
        "note": data.get("note", ""),
    }


# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("Dataset")
    # davis_root_in = st.text_input("DAVIS root", r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS")
    davis_root_in = st.text_input("DAVIS root", r"D:\uwb thesis\RelatedData\train\train")

    # sam_results_root_in = st.text_input("SAM_results root", r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results")
    sam_results_root_in = st.text_input("SAM_results root", r"D:\uwb thesis\RelatedData\train\train\SAM_results_singleobj")
    resolution = st.selectbox("Resolution", ["480p", "1080p"], index=0)

    try:
        davis_root = find_davis_root(davis_root_in)
        seqs = list_sequences(davis_root, resolution)
    except Exception as e:
        st.error(f"Bad DAVIS root: {e}")
        st.stop()

    if not seqs:
        st.error(f"No sequences found under JPEGImages/{resolution}")
        st.stop()

    seq = st.selectbox("Sequence", seqs, index=0)
    frames = list_frames(davis_root, resolution, seq)
    if not frames:
        st.error("No frames found.")
        st.stop()

    st.header("Score hyperparams (sliders)")

    # weights sliders
    wA = st.slider("wA (area)", 0.0, 1.0, 0.25, 0.01)
    wC = st.slider("wC (center)", 0.0, 1.0, 0.25, 0.01)
    wE = st.slider("wE (border)", 0.0, 1.0, 0.25, 0.01)
    wS = st.slider("wSil (silhouette)", 0.0, 1.0, 0.25, 0.01)
    auto_norm = st.checkbox("Normalize weights to sum=1", value=True)

    q_border = st.slider("q_border (E quantile)", 0.0, 1.0, 0.40, 0.01)
    t_area = st.slider("t_area (A target)", 0.0, 1.0, 0.30, 0.01)

    st.header("View")
    frame_idx = st.slider("Frame index", 0, len(frames) - 1, 0, 1)

    compute_seq_mean = st.checkbox("Compute mean IoU over all frames (slow)", value=False)


# =========================
# Resolve paths for current frame
# =========================
frame_file = frames[frame_idx]
stem = Path(frame_file).stem

img_fp = str(Path(davis_root) / "JPEGImages" / resolution / seq / frame_file)

seq_out_dir = Path(sam_results_root_in) / seq
result_fp = seq_out_dir / "json" / f"{stem}_result.json"
gt_json_fp = seq_out_dir / "gt_json" / f"{stem}_gt.json"
gt_png_fp = Path(davis_root) / "Annotations" / resolution / seq / f"{stem}.png"  # typical

if not result_fp.exists():
    st.error(f"Missing result json: {result_fp}")
    st.stop()

# =========================
# Compute ScoreBest for current frame
# =========================
w = np.array([wA, wC, wE, wS], dtype=float)
w = normalize_weights(w) if auto_norm else w

feat_pack = precompute_features_for_frame(str(result_fp), q_border=q_border, t_area=t_area)
X = feat_pack["X"]  # (K,4)
rles = feat_pack["rles"]
ids = feat_pack["ids"]
sam_scores = feat_pack["sam_scores"]

# Load GT (prefer gt_json, fallback DAVIS)
gt_mask = load_gt_from_gtjson(str(gt_json_fp))
if gt_mask is None:
    gt_mask = load_gt_from_davis_png(str(gt_png_fp))

img_rgb = load_rgb_image(img_fp)

if gt_mask is None:
    st.warning("GT not found (neither gt_json nor DAVIS Annotations). IoU cannot be computed.")
    gt_mask = np.zeros(img_rgb.shape[:2], dtype=bool)

H, W = img_rgb.shape[:2]

# Choose best by score
if X.shape[0] == 0:
    st.error("No candidate masks in result.json (annotations empty).")
    st.stop()

scores = X @ w
best_k = int(np.argmax(scores))
best_id = ids[best_k]
best_mask = rle_to_mask_bool(rles[best_k])

# baseline: SAM highest confidence among candidates (if scores exist)
if sam_scores.size == X.shape[0]:
    base_k = int(np.argmax(sam_scores))
    base_id = ids[base_k]
    base_mask = rle_to_mask_bool(rles[base_k])
else:
    base_k = None
    base_id = None
    base_mask = None

# IoU
iou_best = compute_iou(best_mask, gt_mask)
iou_base = compute_iou(base_mask, gt_mask) if base_mask is not None else None

# =========================
# Header: dynamic IoU
# =========================
top_cols = st.columns([2, 2, 6])
top_cols[0].metric("ScoreBest IoU", f"{iou_best:.4f}")
if iou_base is not None:
    top_cols[1].metric("SAM-best IoU", f"{iou_base:.4f}", delta=f"{(iou_best - iou_base):+.4f}")
else:
    top_cols[1].metric("SAM-best IoU", "N/A")

top_cols[2].write(
    f"**Seq:** `{seq}` | **Frame:** `{frame_file}` | "
    f"**Candidates:** {X.shape[0]} | **ScoreBest id:** {best_id} | "
    f"**w:** {w.tolist()} | q={q_border:.2f} t={t_area:.2f} | note={feat_pack.get('note','')}"
)

st.divider()

# =========================
# Main display: GT / Orig / ScoreBest
# =========================
gt_vis = overlay_mask_on_rgb(img_rgb, gt_mask, alpha=0.45)
best_vis = overlay_mask_on_rgb(img_rgb, best_mask, alpha=0.45)

col1, col2, col3 = st.columns(3)
col1.image(gt_vis, caption="GT (overlay)", use_container_width=True)
col2.image(img_rgb, caption="Original", use_container_width=True)
col3.image(best_vis, caption="ScoreBest (overlay, dynamic)", use_container_width=True)

# Optional: also show SAM-best overlay
if base_mask is not None:
    st.markdown("#### Baseline (SAM highest-confidence among saved candidates)")
    st.image(overlay_mask_on_rgb(img_rgb, base_mask, alpha=0.45),
             caption=f"SAM-best overlay | id={base_id}", use_container_width=True)

# =========================
# Optional: sequence mean IoU (slow)
# =========================
if compute_seq_mean:
    st.markdown("### Sequence mean IoU (may take time)")
    ious = []
    base_ious = []
    prog = st.progress(0.0)

    for i, ff in enumerate(frames):
        s = Path(ff).stem
        rf = seq_out_dir / "json" / f"{s}_result.json"
        if not rf.exists():
            continue

        pack = precompute_features_for_frame(str(rf), q_border=q_border, t_area=t_area)
        X2 = pack["X"]
        if X2.shape[0] == 0:
            continue

        # load img size for GT
        # GT prefer gt_json else DAVIS png
        gt2 = load_gt_from_gtjson(str(seq_out_dir / "gt_json" / f"{s}_gt.json"))
        if gt2 is None:
            gt2 = load_gt_from_davis_png(str(Path(davis_root) / "Annotations" / resolution / seq / f"{s}.png"))
        if gt2 is None:
            continue

        rles2 = pack["rles"]
        ids2 = pack["ids"]
        sam2 = pack["sam_scores"]

        sc = X2 @ w
        k2 = int(np.argmax(sc))
        m2 = rle_to_mask_bool(rles2[k2])
        ious.append(compute_iou(m2, gt2))

        if sam2.size == X2.shape[0]:
            kb = int(np.argmax(sam2))
            mb = rle_to_mask_bool(rles2[kb])
            base_ious.append(compute_iou(mb, gt2))

        prog.progress((i + 1) / len(frames))

    if ious:
        mean_iou = float(np.mean(ious))
        st.metric("Mean IoU (ScoreBest)", f"{mean_iou:.4f}")
        if base_ious:
            st.metric("Mean IoU (SAM-best)", f"{float(np.mean(base_ious)):.4f}",
                      delta=f"{(float(np.mean(ious)) - float(np.mean(base_ious))):+.4f}")
    else:
        st.warning("No usable frames for mean IoU computation (missing GT or candidates).")

st.caption("Tips: 如果你发现每帧候选只有3个，通常意味着 DINO 每帧只给了 1 个 box 且 SAM2 multimask 数 M=3。想更多候选可降低阈值/保留更多 box 或做多轮候选池。")