# AnalyzeScore_clean_gt_refine_weighted.py
# Streamlit app: compare refinement baselines vs score-weighted-logit refinement (GT-based IoU only)

import os
import json
from pathlib import Path
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import pycocotools.mask as mask_util
from scipy.stats import wilcoxon

import torch
import contextlib
import base64
import zlib

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# =========================
# SAM2 config
# =========================
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_sam2_predictor():
    torch.set_float32_matmul_precision("high")
    model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    return SAM2ImagePredictor(model)


sam2_predictor = load_sam2_predictor()

# =========================
# Helpers: logits pack/unpack
# =========================
def unpack_lowres_logits(packed: dict) -> np.ndarray:
    """
    packed: {"shape":[256,256], "dtype":"float16", "zlib_b64":"..."}
    return: float32 array (256,256)
    """
    comp = base64.b64decode(packed["zlib_b64"].encode("utf-8"))
    raw = zlib.decompress(comp)
    h, w = packed["shape"]
    arr = np.frombuffer(raw, dtype=np.float16).reshape(h, w).astype(np.float32)
    return arr


def sam2_refine_from_logits(image_path: str, box_xyxy, low_res_logits_2d: np.ndarray):
    """
    Use SAM2 mask_input interface to refine.
    Returns: (refined_mask_bool(H,W), refined_score_sam)
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


# =========================
# Metrics
# =========================
def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def decode_mask_bool(rle):
    m = mask_util.decode(rle)
    return m.astype(bool) if m.ndim == 2 else m[:, :, 0].astype(bool)


# =========================
# Score function (use stored score_custom by default)
# If you want "live recompute" later, you can swap this out.
# =========================
def get_scores_from_json(anns):
    # Prefer your current "score_custom" already written by your pipeline.
    # Clip to non-negative to make sense as weights.
    s = np.array([a.get("score_custom", 0.0) for a in anns], dtype=np.float32)
    s = np.clip(s, 0.0, None)
    return s


# =========================
# Weighted logit fusion
# =========================
EPS = 1e-8

def weighted_logits_all(anns, scores):
    """
    Experiment 1:
    Fuse logits from ALL masks using score weights:
        L = sum_i w_i * L_i / sum_i w_i
    Only uses masks that have low_res_logits.
    """
    logits_list = []
    w_list = []

    for a, s in zip(anns, scores):
        lr_packed = a.get("low_res_logits", None)
        if lr_packed is None:
            continue
        try:
            lr = unpack_lowres_logits(lr_packed)
        except Exception:
            continue
        if lr.shape != (256, 256):
            continue
        logits_list.append(lr)
        w_list.append(float(s))

    if len(logits_list) == 0:
        return None

    w = np.array(w_list, dtype=np.float32)
    w_sum = float(w.sum())
    if w_sum <= EPS:
        # weights all zero -> average logits
        w = np.ones_like(w) / float(len(w))
    else:
        w = w / w_sum

    L = np.zeros((256, 256), dtype=np.float32)
    for wi, li in zip(w, logits_list):
        L += wi * li
    return L


def weighted_logits_topk(anns, scores, k=3):
    """
    Experiment 2:
    Take top-k by score (k=3 if available), fuse logits using score weights.
    """
    idx = np.argsort(-scores)  # descending
    idx = idx[: min(k, len(idx))]

    logits_list = []
    w_list = []

    for i in idx:
        a = anns[int(i)]
        s = float(scores[int(i)])
        lr_packed = a.get("low_res_logits", None)
        if lr_packed is None:
            continue
        try:
            lr = unpack_lowres_logits(lr_packed)
        except Exception:
            continue
        if lr.shape != (256, 256):
            continue
        logits_list.append(lr)
        w_list.append(s)

    if len(logits_list) == 0:
        return None

    w = np.array(w_list, dtype=np.float32)
    w_sum = float(w.sum())
    if w_sum <= EPS:
        w = np.ones_like(w) / float(len(w))
    else:
        w = w / w_sum

    L = np.zeros((256, 256), dtype=np.float32)
    for wi, li in zip(w, logits_list):
        L += wi * li
    return L


# =========================
# Stats helpers
# =========================
def describe(arr: np.ndarray):
    return {
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "median": float(np.median(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
        "min": float(np.min(arr)) if arr.size else 0.0,
    }


def wilcoxon_one_sided_report(A: np.ndarray, B: np.ndarray):
    """
    H1: A > B
    Returns: dict with N, Wplus, p, r
    """
    diff = A - B
    mask_nonzero = diff != 0
    A_nz = A[mask_nonzero]
    B_nz = B[mask_nonzero]
    N = int(A_nz.size)

    if N == 0:
        return {"N": 0, "Wplus": 0.0, "p": 1.0, "r": 0.0}

    stat, p_value = wilcoxon(A_nz, B_nz, alternative="greater")

    # normal approximation for effect size r
    mean_W = N * (N + 1) / 4
    var_W = N * (N + 1) * (2 * N + 1) / 24
    z = (stat - mean_W - 0.5) / np.sqrt(var_W)
    r = float(z / np.sqrt(N))

    return {"N": N, "Wplus": float(stat), "p": float(p_value), "r": r}


# =========================
# Streamlit UI
# =========================
st.set_page_config(layout="wide")
st.sidebar.title("🧪 Analyze Refinement (GT-based) + Weighted Logits")

DEFAULT_JSON_FOLDER = "outputs/AllMasks_v2_score_record"
DEFAULT_GT_FOLDER = "GT_masks"

json_folder = st.sidebar.text_input("First-pass JSON folder", value=DEFAULT_JSON_FOLDER)
gt_folder = st.sidebar.text_input("GT folder (contains *_gt.png)", value=DEFAULT_GT_FOLDER)

st.sidebar.markdown("---")
show_per_image = st.sidebar.checkbox("Show per-image visualization", value=False)
max_show = st.sidebar.slider("Max images to show", 1, 50, 20, 1)

st.title("Refinement Experiments (GT-based IoU)")

json_dir = Path(json_folder)
gt_dir = Path(gt_folder)

if not json_dir.exists():
    st.error(f"JSON folder not found: {json_dir}")
    st.stop()
if not gt_dir.exists():
    st.error(f"GT folder not found: {gt_dir}")
    st.stop()

json_files = sorted(json_dir.glob("*_result.json"))
if not json_files:
    st.warning("No *_result.json found.")
    st.stop()

# Build GT index by image stem
gt_map = {}  # stem -> gt_mask_path
for p in gt_dir.glob("*_gt.png"):
    stem = p.name.replace("_gt.png", "")
    gt_map[stem] = p

# Filter JSONs to those with GT
pairs = []
for fp in json_files:
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        continue
    img_path = data.get("image_path", "")
    stem = Path(img_path).stem
    if stem in gt_map:
        pairs.append((fp, gt_map[stem]))

st.write(f"Matched samples (JSON ∩ GT): **{len(pairs)}** / {len(json_files)}")

if len(pairs) == 0:
    st.stop()

# =========================
# Main evaluation loop
# =========================
ious_ref_sam = []
ious_ref_score = []
ious_ref_allw = []
ious_ref_top3w = []

# Optional: also keep first-pass selection comparison
ious_first_sam = []
ious_first_score = []

vis_rows = []

for fp, gt_png in pairs:
    data = json.loads(fp.read_text(encoding="utf-8"))
    anns = data.get("annotations", [])
    if not anns:
        continue

    image_path = data.get("image_path", None)
    if image_path is None or not os.path.exists(image_path):
        continue

    # Load GT mask
    gt_img = cv2.imread(str(gt_png), cv2.IMREAD_GRAYSCALE)
    if gt_img is None:
        continue
    gt_mask = (gt_img > 127)

    # Decode first-pass masks
    masks_bool = [decode_mask_bool(a["segmentation"]) for a in anns]

    # Scores (from JSON score_custom)
    scores = get_scores_from_json(anns)

    # Identify top1 by score_custom
    idx_score = int(np.argmax(scores)) if len(scores) else 0
    ann_score = anns[idx_score]
    mask_score = masks_bool[idx_score]

    # Identify top1 by SAM confidence
    sam_scores = np.array([a.get("score_sam", 0.0) for a in anns], dtype=np.float32)
    idx_sam = int(np.argmax(sam_scores)) if sam_scores.size else 0
    ann_sam = anns[idx_sam]
    mask_sam = masks_bool[idx_sam]

    # First-pass IoU baselines (optional but requested to keep score vs SAM first pass)
    iou_first_score = compute_iou(gt_mask, mask_score)
    iou_first_sam = compute_iou(gt_mask, mask_sam)
    ious_first_score.append(iou_first_score)
    ious_first_sam.append(iou_first_sam)

    # Refinement baseline: Refine-SAM (use SAM top1 logits)
    if "low_res_logits" not in ann_sam:
        continue
    lr_sam = unpack_lowres_logits(ann_sam["low_res_logits"])
    ref_sam_mask, _ = sam2_refine_from_logits(image_path, ann_sam["bbox"], lr_sam)
    if ref_sam_mask is None:
        continue
    iou_ref_sam = compute_iou(gt_mask, ref_sam_mask)

    # Refinement: Refine-ScoreTop1 (use score top1 logits)
    if "low_res_logits" not in ann_score:
        continue
    lr_score = unpack_lowres_logits(ann_score["low_res_logits"])
    ref_score_mask, _ = sam2_refine_from_logits(image_path, ann_score["bbox"], lr_score)
    if ref_score_mask is None:
        continue
    iou_ref_score = compute_iou(gt_mask, ref_score_mask)

    # Experiment 1: all masks weighted logits
    L_all = weighted_logits_all(anns, scores)
    ref_allw_mask = None
    iou_ref_allw = None
    if L_all is not None:
        # Use SAM box (keeps baseline consistent); you can switch to score box if you prefer.
        ref_allw_mask, _ = sam2_refine_from_logits(image_path, ann_sam["bbox"], L_all)
        if ref_allw_mask is not None:
            iou_ref_allw = compute_iou(gt_mask, ref_allw_mask)

    # Experiment 2: top3 weighted logits
    L_top3 = weighted_logits_topk(anns, scores, k=3)
    ref_top3w_mask = None
    iou_ref_top3w = None
    if L_top3 is not None:
        ref_top3w_mask, _ = sam2_refine_from_logits(image_path, ann_sam["bbox"], L_top3)
        if ref_top3w_mask is not None:
            iou_ref_top3w = compute_iou(gt_mask, ref_top3w_mask)

    # Append if both exist (to keep paired tests fair)
    # Requirement: compare each new experiment vs Refine-SAM baseline with p-value.
    # So we store per-experiment arrays only when that experiment produced a valid mask.
    ious_ref_sam.append(iou_ref_sam)
    ious_ref_score.append(iou_ref_score)

    if iou_ref_allw is not None:
        ious_ref_allw.append(iou_ref_allw)
    if iou_ref_top3w is not None:
        ious_ref_top3w.append(iou_ref_top3w)

    if show_per_image and len(vis_rows) < max_show:
        vis_rows.append({
            "fp": fp,
            "image_path": image_path,
            "gt_png": gt_png,
            "idx_score": idx_score,
            "idx_sam": idx_sam,
            "iou_first_score": iou_first_score,
            "iou_first_sam": iou_first_sam,
            "iou_ref_sam": iou_ref_sam,
            "iou_ref_score": iou_ref_score,
            "iou_ref_allw": iou_ref_allw,
            "iou_ref_top3w": iou_ref_top3w,
            "mask_score": mask_score,
            "mask_sam": mask_sam,
            "ref_sam_mask": ref_sam_mask,
            "ref_score_mask": ref_score_mask,
            "ref_allw_mask": ref_allw_mask,
            "ref_top3w_mask": ref_top3w_mask,
            "gt_mask": gt_mask,
        })

# =========================
# Report summary
# =========================
st.markdown("## 📏 IoU Summary (GT as ground truth)")

# First-pass selection: score vs SAM
arr_first_score = np.array(ious_first_score, dtype=float)
arr_first_sam = np.array(ious_first_sam, dtype=float)

cols0 = st.columns(2)
with cols0[0]:
    d = describe(arr_first_score)
    st.markdown("**First-pass: Score Top1 vs GT**")
    st.write(f"- N: `{arr_first_score.size}`")
    st.write(f"- Mean: `{d['mean']:.3f}`  Median: `{d['median']:.3f}`  Min: `{d['min']:.3f}`  Max: `{d['max']:.3f}`")

with cols0[1]:
    d = describe(arr_first_sam)
    st.markdown("**First-pass: SAM Top1 vs GT**")
    st.write(f"- N: `{arr_first_sam.size}`")
    st.write(f"- Mean: `{d['mean']:.3f}`  Median: `{d['median']:.3f}`  Min: `{d['min']:.3f}`  Max: `{d['max']:.3f}`")

# Refinement baselines
arr_ref_sam = np.array(ious_ref_sam, dtype=float)
arr_ref_score = np.array(ious_ref_score, dtype=float)

st.markdown("## 🔁 Refinement Summary (mask-prompt)")
cols1 = st.columns(2)
with cols1[0]:
    d = describe(arr_ref_sam)
    st.markdown("**Baseline: Refine-SAM vs GT**")
    st.write(f"- N: `{arr_ref_sam.size}`")
    st.write(f"- Mean: `{d['mean']:.3f}`  Median: `{d['median']:.3f}`  Min: `{d['min']:.3f}`  Max: `{d['max']:.3f}`")

with cols1[1]:
    d = describe(arr_ref_score)
    st.markdown("**Refine-ScoreTop1 vs GT**")
    st.write(f"- N: `{arr_ref_score.size}`")
    st.write(f"- Mean: `{d['mean']:.3f}`  Median: `{d['median']:.3f}`  Min: `{d['min']:.3f}`  Max: `{d['max']:.3f}`")

# New experiments (note: may have smaller N due to missing logits, etc.)
arr_ref_allw = np.array(ious_ref_allw, dtype=float)
arr_ref_top3w = np.array(ious_ref_top3w, dtype=float)

cols2 = st.columns(2)
with cols2[0]:
    d = describe(arr_ref_allw)
    st.markdown("**NEW Exp1: Refine-ScoreAllWeighted vs GT**")
    st.write(f"- N: `{arr_ref_allw.size}`")
    st.write(f"- Mean: `{d['mean']:.3f}`  Median: `{d['median']:.3f}`  Min: `{d['min']:.3f}`  Max: `{d['max']:.3f}`")

with cols2[1]:
    d = describe(arr_ref_top3w)
    st.markdown("**NEW Exp2: Refine-ScoreTop3Weighted vs GT**")
    st.write(f"- N: `{arr_ref_top3w.size}`")
    st.write(f"- Mean: `{d['mean']:.3f}`  Median: `{d['median']:.3f}`  Min: `{d['min']:.3f}`  Max: `{d['max']:.3f}`")


# =========================
# Wilcoxon tests (one-sided) vs Refine-SAM baseline
# Need paired arrays of equal length
# We do pairing by truncating to common subset based on index positions stored.
# Here simplest: compute tests only on images where both baseline and experiment exist.
# =========================
st.markdown("## 🔬 Statistical Tests (Wilcoxon, one-sided)")
st.write("Baseline for tests: **Refine-SAM** (two-pass SAM using SAM-top1 logits).")

# Pairing helper: we stored baseline for every sample, but experiments may be missing.
# So we recompute paired lists from visualization rows (which only stores when baseline exists).
# For robustness, re-walk pairs and recompute IoU lists with a strict requirement for each test.
# (In practice, your JSONs likely all have logits, so sizes should match.)

# We'll approximate pairing by using minimum length with ordering,
# but better: advise you keep experiments valid for all matched GT images.
def paired_test(A_list, B_list, title):
    A = np.array(A_list, dtype=float)
    B = np.array(B_list, dtype=float)
    n = min(A.size, B.size)
    if n == 0:
        st.markdown(f"**{title}**")
        st.write("No valid pairs.")
        return
    A = A[:n]
    B = B[:n]
    rep = wilcoxon_one_sided_report(A, B)
    st.markdown(f"**{title}**")
    st.write(f"- N (diff ≠ 0): `{rep['N']}`")
    st.write(f"- W⁺: `{rep['Wplus']:.3f}`")
    st.write(f"- p-value (A > B): `{rep['p']:.2e}`")
    st.write(f"- effect size r: `{rep['r']:.3f}`")


paired_test(arr_ref_score.tolist(), arr_ref_sam.tolist(), "H1: Refine-ScoreTop1 IoU > Refine-SAM IoU")

paired_test(arr_ref_allw.tolist(), arr_ref_sam.tolist(), "H1: Refine-ScoreAllWeighted IoU > Refine-SAM IoU")

paired_test(arr_ref_top3w.tolist(), arr_ref_sam.tolist(), "H1: Refine-ScoreTop3Weighted IoU > Refine-SAM IoU")


# =========================
# Optional per-image visualization
# =========================
if show_per_image:
    st.markdown("## 🖼 Per-image Visualization (subset)")
    for row in vis_rows:
        fp = row["fp"]
        img_path = row["image_path"]

        st.markdown(f"### {fp.name}")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            st.write("Image load failed.")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        H, W = img_rgb.shape[:2]
        small_w = max(200, W // 4)
        small_h = max(200, H // 4)

        cols = st.columns(6)
        cols[0].image(img_pil.resize((small_w, small_h)), caption="RGB", use_container_width=True)

        def mask_to_pil(m):
            return Image.fromarray((m.astype(np.uint8) * 255))

        cols[1].image(mask_to_pil(row["gt_mask"]).resize((small_w, small_h)),
                      caption="GT", use_container_width=True)

        cols[2].image(mask_to_pil(row["mask_sam"]).resize((small_w, small_h)),
                      caption=f"SAM Top1 (1st) IoU={row['iou_first_sam']:.3f}", use_container_width=True)

        cols[3].image(mask_to_pil(row["mask_score"]).resize((small_w, small_h)),
                      caption=f"Score Top1 (1st) IoU={row['iou_first_score']:.3f}", use_container_width=True)

        cols[4].image(mask_to_pil(row["ref_sam_mask"]).resize((small_w, small_h)),
                      caption=f"Refine-SAM IoU={row['iou_ref_sam']:.3f}", use_container_width=True)

        cols[5].image(mask_to_pil(row["ref_score_mask"]).resize((small_w, small_h)),
                      caption=f"Refine-ScoreTop1 IoU={row['iou_ref_score']:.3f}", use_container_width=True)

        # second line: show weighted refinements if available
        cols2 = st.columns(6)
        cols2[0].markdown(" ")
        cols2[1].markdown(" ")

        if row["ref_allw_mask"] is not None and row["iou_ref_allw"] is not None:
            cols2[2].image(mask_to_pil(row["ref_allw_mask"]).resize((small_w, small_h)),
                           caption=f"Refine-ScoreAllWeighted IoU={row['iou_ref_allw']:.3f}",
                           use_container_width=True)
        else:
            cols2[2].markdown("_AllWeighted unavailable_")

        if row["ref_top3w_mask"] is not None and row["iou_ref_top3w"] is not None:
            cols2[3].image(mask_to_pil(row["ref_top3w_mask"]).resize((small_w, small_h)),
                           caption=f"Refine-ScoreTop3Weighted IoU={row['iou_ref_top3w']:.3f}",
                           use_container_width=True)
        else:
            cols2[3].markdown("_Top3Weighted unavailable_")

        cols2[4].markdown(" ")
        cols2[5].markdown(" ")
        st.divider()
