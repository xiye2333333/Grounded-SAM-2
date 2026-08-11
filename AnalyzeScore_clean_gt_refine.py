import os
import json
from pathlib import Path
import base64, zlib, contextlib

import numpy as np
import cv2
import streamlit as st
from PIL import Image
import pycocotools.mask as mask_util
from scipy.stats import wilcoxon

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ==============================
# SAM2 Config
# ==============================
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_sam2_predictor():
    torch.set_float32_matmul_precision("high")
    model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    return SAM2ImagePredictor(model)


sam2_predictor = load_sam2_predictor()


# ==============================
# Utilities
# ==============================
def unpack_lowres_logits(packed: dict) -> np.ndarray:
    """
    packed: {"shape":[256,256], "dtype":"float16", "zlib_b64": "..."}
    return: float32 array (256,256)
    """
    comp = base64.b64decode(packed["zlib_b64"].encode("utf-8"))
    raw = zlib.decompress(comp)
    h, w = packed["shape"]
    # dtype in json is "float16" in your example
    arr = np.frombuffer(raw, dtype=np.float16).reshape(h, w).astype(np.float32)
    return arr


def sam2_refine_from_logits(image_path: str, box_xyxy, low_res_logits_2d: np.ndarray):
    """
    Use mask_input (low-res logits) + same box to refine.
    Returns:
        refined_mask_bool: (H,W) bool
        refined_score_sam: float
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, 0.0

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    sam2_predictor.set_image(img_rgb)

    mask_input = low_res_logits_2d[None, :, :]  # (1,256,256)
    box_in = np.array(box_xyxy, dtype=np.float32)[None, :]  # (1,4)

    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if DEVICE.startswith("cuda")
        else contextlib.nullcontext()
    )
    with ctx:
        masks2, scores2, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_in,
            mask_input=mask_input,
            multimask_output=False,
        )

    m = masks2
    if m is None:
        return None, 0.0
    if m.ndim == 4:  # (1,1,H,W)
        m = m[:, 0, :, :]
    refined = m[0].astype(bool)
    refined_score = float(np.array(scores2).reshape(-1)[0]) if scores2 is not None else 0.0
    return refined, refined_score


@st.cache_data(show_spinner=False)
def cached_refine(image_path: str, box_xyxy, packed_logits: dict):
    lr = unpack_lowres_logits(packed_logits)
    return sam2_refine_from_logits(image_path, box_xyxy, lr)


def decode_mask_bool(rle):
    m = mask_util.decode(rle)
    return m.astype(bool) if m.ndim == 2 else m[:, :, 0].astype(bool)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def load_gt_mask_bool(gt_png_path: Path) -> np.ndarray | None:
    m = cv2.imread(str(gt_png_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 127)


def stem_from_result_json(fp: Path) -> str:
    # xxx_result.json -> xxx
    name = fp.name
    return name[:-len("_result.json")] if name.endswith("_result.json") else fp.stem


# ==============================
# Scoring (your latest)
# ==============================
MIN_AREA_RATIO_DEFAULT = 0.00
EPS = 1e-8


def compute_distance_transform(h, w):
    border_mask = np.zeros((h, w), np.uint8)
    border_mask[1:-1, 1:-1] = 1
    return cv2.distanceTransform(border_mask, distanceType=cv2.DIST_L2, maskSize=5)


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


def area_term_parabola(x: float, d: float, eps: float = 1e-6) -> float:
    d = float(np.clip(d, eps, 1.0))
    x = float(np.clip(x, 0.0, 1.0))
    val = 1.0 - ((x - d) / d) ** 2
    return float(np.clip(val, 0.0, 1.0))


def normalize_weights(w_area, w_center, w_border, w_sil, eps: float = 1e-8):
    s = float(w_area + w_center + w_border + w_sil)
    if s <= eps:
        return 0.25, 0.25, 0.25, 0.25
    return w_area / s, w_center / s, w_border / s, w_sil / s


def compute_scores_new(masks_bool, W, H, W_AREA, W_CENTER, W_BORDER, W_SIL, Q_BORDER, t_area, MIN_AREA_RATIO):
    img_area = W * H
    cx, cy = W / 2, H / 2
    dist_map = compute_distance_transform(H, W)
    d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

    wA, wC, wB, wS = normalize_weights(W_AREA, W_CENTER, W_BORDER, W_SIL)

    scores = []
    for seg in masks_bool:
        area_px = int(seg.sum())
        if area_px < MIN_AREA_RATIO * img_area:
            scores.append(0.0)
            continue

        A_raw = area_px / img_area
        A = area_term_parabola(A_raw, t_area)

        ys, xs = np.where(seg)
        if xs.size == 0:
            scores.append(0.0)
            continue
        mx, my = xs.mean(), ys.mean()
        Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
        C = 1.0 - float(np.clip(Dp, 0.0, 1.0))

        q = float(np.quantile(dist_map[seg], Q_BORDER)) if np.any(seg) else 0.0
        E = float(np.clip(q / d_max, 0.0, 1.0))

        Sil = compute_silhouette_score_v2(seg)

        S = wA * A + wC * C + wB * E + wS * Sil
        scores.append(float(np.clip(S, 0.0, 1.0)))

    return scores


# ==============================
# Streamlit UI
# ==============================
st.set_page_config(layout="wide")
st.sidebar.title("🧪 Analyze (GT-based)")

AVAILABLE_FOLDERS = [
    "outputs/AllMasks_v2_score_record",
    "outputs/AdjustWeightData",
    "outputs/CatagroiesAnalyze/BackgroundNoise",
    "outputs/CatagroiesAnalyze/FrontNoise",
    "outputs/CatagroiesAnalyze/MutiObject",
    "outputs/CatagroiesAnalyze/SepratePieces",
    "outputs/CatagroiesAnalyze/ShallowAndHoles",
    "outputs/score_fail_case",
]
folder = st.sidebar.selectbox("📂 JSON Folder", AVAILABLE_FOLDERS, index=0)

GT_DIR = st.sidebar.text_input("📌 GT folder", value="GT_masks")

st.sidebar.markdown("---")
W_AREA = st.sidebar.slider("W_AREA", 0.0, 1.0, 0.2, 0.05)
W_CENTER = st.sidebar.slider("W_CENTER", 0.0, 1.0, 0.2, 0.05)
W_BORDER = st.sidebar.slider("W_BORDER", 0.0, 1.0, 0.6, 0.05)
W_SIL = st.sidebar.slider("W_SIL", 0.0, 1.0, 0.2, 0.05)
Q_BORDER = st.sidebar.slider("Q_BORDER", 0.0, 1.0, 0.25, 0.01)
t_area = st.sidebar.slider("🎯 t_area", 0.0, 1.0, 0.4, 0.005)
MIN_AREA_RATIO = st.sidebar.slider("MIN_AREA_RATIO", 0.0, 0.1, 0.01, 0.005)

st.sidebar.markdown("---")
show_per_image = st.sidebar.checkbox("Show per-image visualization", value=True)
max_show = st.sidebar.slider("Max images to show", 1, 200, 30, 1)

st.title("GT-based Mask Analysis + Two-way Refinement (score vs SAM logits)")

# ==============================
# Load & filter by GT overlap
# ==============================
folder_path = Path(folder)
if not folder_path.exists():
    st.error(f"JSON folder not found: {folder_path}")
    st.stop()

gt_dir = Path(GT_DIR)
if not gt_dir.exists():
    st.error(f"GT folder not found: {gt_dir}")
    st.stop()

json_files_all = sorted(folder_path.glob("*_result.json"))
gt_stems = {p.name[:-len("_gt.png")] for p in gt_dir.glob("*_gt.png") if p.name.endswith("_gt.png")}
json_files = [fp for fp in json_files_all if stem_from_result_json(fp) in gt_stems]

st.sidebar.write(f"JSON total: {len(json_files_all)}")
st.sidebar.write(f"GT masks: {len(gt_stems)}")
st.sidebar.write(f"Overlap used: {len(json_files)}")

if not json_files:
    st.warning("No overlapped samples found (need xxx_result.json AND GT_masks/xxx_gt.png).")
    st.stop()


# ==============================
# Main evaluation loop
# ==============================
LOGITS_KEY = "low_res_logits"  # confirmed by your example JSON

ious_score_top1 = []
ious_sam_top1 = []
ious_refine_score = []
ious_refine_sam = []

skipped_missing_logits = 0
skipped_refine_fail = 0
skipped_image_read = 0

# For visualization cache
per_image_rows = []

for fp in json_files:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    anns = data.get("annotations", [])
    if not anns:
        continue

    stem = stem_from_result_json(fp)
    gt_path = gt_dir / f"{stem}_gt.png"
    gt_mask = load_gt_mask_bool(gt_path)
    if gt_mask is None:
        continue

    H, W = int(data["img_height"]), int(data["img_width"])
    masks_bool = [decode_mask_bool(a["segmentation"]) for a in anns]

    # -------- First-pass selection: score-top1 ----------
    score_vals = compute_scores_new(
        masks_bool, W, H,
        W_AREA=W_AREA, W_CENTER=W_CENTER, W_BORDER=W_BORDER, W_SIL=W_SIL,
        Q_BORDER=Q_BORDER, t_area=t_area, MIN_AREA_RATIO=MIN_AREA_RATIO
    )
    idx_score = int(np.argmax(score_vals))
    mask_score = masks_bool[idx_score]
    iou_score = compute_iou(gt_mask, mask_score)

    # -------- First-pass baseline: SAM-top1 (score_sam) ----------
    sam_scores = [a.get("score_sam", 0.0) for a in anns]
    idx_sam = int(np.argmax(sam_scores)) if sam_scores else idx_score
    mask_sam = masks_bool[idx_sam]
    iou_sam = compute_iou(gt_mask, mask_sam)

    # -------- Refinement paths ----------
    ann_score = anns[idx_score]
    ann_sam = anns[idx_sam]

    if (LOGITS_KEY not in ann_score) or (LOGITS_KEY not in ann_sam):
        skipped_missing_logits += 1
        continue

    # use cached refine
    refined_score_mask, refined_score_sam_score = cached_refine(
        data["image_path"], ann_score["bbox"], ann_score[LOGITS_KEY]
    )
    refined_sam_mask, refined_sam_sam_score = cached_refine(
        data["image_path"], ann_sam["bbox"], ann_sam[LOGITS_KEY]
    )

    if refined_score_mask is None or refined_sam_mask is None:
        skipped_refine_fail += 1
        continue

    iou_ref_score = compute_iou(gt_mask, refined_score_mask)
    iou_ref_sam = compute_iou(gt_mask, refined_sam_mask)

    ious_score_top1.append(iou_score)
    ious_sam_top1.append(iou_sam)
    ious_refine_score.append(iou_ref_score)
    ious_refine_sam.append(iou_ref_sam)

    if show_per_image:
        per_image_rows.append({
            "fp": fp,
            "data": data,
            "anns": anns,
            "W": W,
            "H": H,
            "gt_path": gt_path,
            "gt_mask": gt_mask,
            "idx_score": idx_score,
            "idx_sam": idx_sam,
            "score_top1_iou": iou_score,
            "sam_top1_iou": iou_sam,
            "ref_score_iou": iou_ref_score,
            "ref_sam_iou": iou_ref_sam,
            "ref_score_sam": refined_score_sam_score,
            "ref_sam_sam": refined_sam_sam_score,
            "score_val_top1": float(score_vals[idx_score]),
            "sam_val_top1": float(sam_scores[idx_sam]) if sam_scores else 0.0,
        })


arr_score = np.array(ious_score_top1, dtype=float)
arr_sam = np.array(ious_sam_top1, dtype=float)
arr_ref_score = np.array(ious_refine_score, dtype=float)
arr_ref_sam = np.array(ious_refine_sam, dtype=float)

st.markdown("### 📌 Dataset Summary (GT-overlap only)")
st.write(f"Valid evaluated samples: **{len(arr_score)}**")
st.write(f"Skipped (missing logits): **{skipped_missing_logits}**")
st.write(f"Skipped (refine failed): **{skipped_refine_fail}**")

if len(arr_score) == 0:
    st.stop()

# ==============================
# Aggregate Stats
# ==============================
def stat_block(title: str, x: np.ndarray):
    st.markdown(f"**{title}**")
    st.write(f"- Mean: `{float(np.mean(x)):.3f}`")
    st.write(f"- Median: `{float(np.median(x)):.3f}`")
    st.write(f"- Min/Max: `{float(np.min(x)):.3f}` / `{float(np.max(x)):.3f}`")


cols = st.columns(4)
with cols[0]:
    stat_block("Score Top1 vs GT (first-pass)", arr_score)
with cols[1]:
    stat_block("SAM Top1 vs GT (first-pass)", arr_sam)
with cols[2]:
    stat_block("Refine-score vs GT", arr_ref_score)
with cols[3]:
    stat_block("Refine-sam vs GT", arr_ref_sam)

st.markdown("---")

# ==============================
# Wilcoxon Tests
# ==============================
st.markdown("### 🔬 Statistical Tests (Wilcoxon, one-sided)")

def wilcoxon_report(a: np.ndarray, b: np.ndarray, title: str):
    diff = a - b
    mask_nz = diff != 0
    a_nz, b_nz = a[mask_nz], b[mask_nz]
    N = len(a_nz)
    st.markdown(f"**{title}**")
    if N == 0:
        st.write("- All pairs equal; no non-zero differences.")
        return

    stat, p_value = wilcoxon(a_nz, b_nz, alternative="greater")

    mean_W = N * (N + 1) / 4
    var_W = N * (N + 1) * (2 * N + 1) / 24
    z = (stat - mean_W - 0.5) / np.sqrt(var_W)
    effect_r = z / np.sqrt(N)

    st.write(f"- N (diff ≠ 0): `{N}`")
    st.write(f"- W⁺: `{stat:.3f}`")
    st.write(f"- p-value (A > B): `{p_value:.2e}`")
    st.write(f"- effect size r: `{effect_r:.3f}`")


# old comparison you wanted to keep (first-pass)
wilcoxon_report(arr_score, arr_sam, "H1: Score Top1 IoU > SAM Top1 IoU (first-pass selection)")

# new target: does score help refinement more than sam's own logits selection?
wilcoxon_report(arr_ref_score, arr_ref_sam, "H1: Refine-score IoU > Refine-sam IoU (score helps refinement)")

st.markdown("---")

# ==============================
# Visualization per image (optional)
# ==============================
if show_per_image:
    st.markdown("### 🖼 Per-image Visualization (GT + first-pass + two refinements)")
    show_rows = per_image_rows[:max_show]

    for row in show_rows:
        fp = row["fp"]
        data = row["data"]
        anns = row["anns"]
        W, H = row["W"], row["H"]
        gt_path = row["gt_path"]

        st.markdown(f"#### {fp.name}  |  "
                    f"IoU: score={row['score_top1_iou']:.3f}, sam={row['sam_top1_iou']:.3f}, "
                    f"ref(score)={row['ref_score_iou']:.3f}, ref(sam)={row['ref_sam_iou']:.3f}")

        # load images
        img_bgr = cv2.imread(data["image_path"])
        if img_bgr is None:
            st.warning(f"Cannot read image: {data['image_path']}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        gt_mask = load_gt_mask_bool(gt_path)
        gt_pil = Image.fromarray((gt_mask.astype(np.uint8) * 255))

        idx_score = row["idx_score"]
        idx_sam = row["idx_sam"]

        score_mask = decode_mask_bool(anns[idx_score]["segmentation"])
        sam_mask = decode_mask_bool(anns[idx_sam]["segmentation"])

        score_pil = Image.fromarray((score_mask.astype(np.uint8) * 255))
        sam_pil = Image.fromarray((sam_mask.astype(np.uint8) * 255))

        # refinements
        ref_score_mask, _ = cached_refine(data["image_path"], anns[idx_score]["bbox"], anns[idx_score][LOGITS_KEY])
        ref_sam_mask, _ = cached_refine(data["image_path"], anns[idx_sam]["bbox"], anns[idx_sam][LOGITS_KEY])

        ref_score_pil = Image.fromarray((ref_score_mask.astype(np.uint8) * 255))
        ref_sam_pil = Image.fromarray((ref_sam_mask.astype(np.uint8) * 255))

        cols = st.columns(6)
        cols[0].image(pil_img, caption="RGB", use_container_width=True)
        cols[1].image(gt_pil, caption="GT", use_container_width=True)
        cols[2].image(score_pil, caption=f"Score Top1 (score={row['score_val_top1']:.3f})", use_container_width=True)
        cols[3].image(sam_pil, caption=f"SAM Top1 (sam={row['sam_val_top1']:.3f})", use_container_width=True)
        cols[4].image(ref_score_pil, caption="Refine-score (mask_input=score logits)", use_container_width=True)
        cols[5].image(ref_sam_pil, caption="Refine-sam (mask_input=sam logits)", use_container_width=True)

        st.divider()
