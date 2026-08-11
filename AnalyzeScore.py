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
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_sam2_predictor():
    torch.set_float32_matmul_precision("high")
    model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    return SAM2ImagePredictor(model)

sam2_predictor = load_sam2_predictor()

def unpack_lowres_logits(packed: dict) -> np.ndarray:
    """
    packed: {"shape":[256,256], "dtype":"float16", "zlib_b64": "..."}
    return: float32 array (256,256)
    """
    comp = base64.b64decode(packed["zlib_b64"].encode("utf-8"))
    raw = zlib.decompress(comp)
    h, w = packed["shape"]
    arr = np.frombuffer(raw, dtype=np.float16).reshape(h, w).astype(np.float32)
    return arr

def sam2_refine_from_logits(image_path: str, box_xyxy, low_res_logits_2d: np.ndarray):
    """
    image_path: path to original RGB image (the same used in SAM run)
    box_xyxy: list[4] (xyxy)
    low_res_logits_2d: (256,256) float32 logits
    returns:
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

    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if DEVICE.startswith("cuda") else contextlib.nullcontext()
    with ctx:
        masks2, scores2, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_in,
            mask_input=mask_input,
            multimask_output=False,
        )

    m = masks2
    if m.ndim == 4:  # (1,1,H,W)
        m = m[:, 0, :, :]
    refined = m[0].astype(bool)
    refined_score = float(np.array(scores2).reshape(-1)[0]) if scores2 is not None else 0.0
    return refined, refined_score

# ==============================
# UI 配置
# ==============================
st.set_page_config(layout="wide")
st.sidebar.title("🎛 Mask Scoring UI")

AVAILABLE_FOLDERS = [
    "outputs/AdjustWeightData",
    "outputs/CatagroiesAnalyze/BackgroundNoise",
    "outputs/CatagroiesAnalyze/FrontNoise",
    "outputs/CatagroiesAnalyze/MutiObject",
    "outputs/CatagroiesAnalyze/SepratePieces",
    "outputs/CatagroiesAnalyze/ShallowAndHoles",
    "outputs/AllMasks_v2_score_record",
    "outputs/score_fail_case"
]
folder = st.sidebar.selectbox("📂 Select JSON Folder", AVAILABLE_FOLDERS, index=0)
GT_DIR = st.sidebar.text_input("📌 GT folder (GT_masks)", value="GT_masks")

st.sidebar.markdown("---")

# ✅ 权重调节栏
W_AREA = st.sidebar.slider("W_AREA", 0.0, 1.0, 0.2, 0.05)
W_CENTER = st.sidebar.slider("W_CENTER", 0.0, 1.0, 0.2, 0.05)
W_BORDER = st.sidebar.slider("W_BORDER", 0.0, 1.0, 0.6, 0.05)
W_SIL = st.sidebar.slider("W_SILHOUETTE", 0.0, 1.0, 0.6, 0.05)
Q_BORDER = st.sidebar.slider("Q_BORDER", 0.0, 1.0, 0.40, 0.01)

t_area = st.sidebar.slider("🎯 Area target ratio (t)", 0.0, 1.0, 0.3, 0.005)
k_area = st.sidebar.slider("⚡ Area steepness (k)", 1.0, 30.0, 30.0, 1.0)
sigmoid_isOn = st.sidebar.checkbox("Use Sigmoid for Area Term", value=True)

st.sidebar.markdown("---")
do_write_json = st.sidebar.toggle("写回 JSON: Old=初始分数", value=True)
do_backup = st.sidebar.toggle("写入前 *.bak 备份", value=True)

st.title("🔥 Adjustable Scoring UI ┃ With Human Match Metrics + SAM Visualization")

Q_BORDER_INIT = 0.25
MIN_AREA_RATIO = 0.00


# ======================================================
# 工具函数
# ======================================================
def compute_distance_transform(h, w):
    border_mask = np.zeros((h, w), np.uint8)
    border_mask[1:-1, 1:-1] = 1
    return cv2.distanceTransform(border_mask, distanceType=cv2.DIST_L2, maskSize=5)


def compute_silhouette_score_v2(mask: np.ndarray) -> float:
    mask_u8 = (mask.astype(np.uint8) > 0).astype(np.uint8)

    # 1) Solidity
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

    # 2) Fragmentation (ignore tiny components)
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

    # 3) Hole ratio via floodfill on background
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

EPS = 1e-8

def area_term_parabola(x: float, d: float, eps: float = 1e-6) -> float:
    """
    Piecewise parabola peak at d:
    f(x)=max(0, 1-((x-d)/d)^2), clipped to [0,1]
    """
    d = float(np.clip(d, eps, 1.0))
    x = float(np.clip(x, 0.0, 1.0))
    val = 1.0 - ((x - d) / d) ** 2
    return float(np.clip(val, 0.0, 1.0))

def normalize_weights(w_area, w_center, w_border, w_sil, eps: float = 1e-8):
    s = float(w_area + w_center + w_border + w_sil)
    if s <= eps:
        return 0.25, 0.25, 0.25, 0.25
    return w_area / s, w_center / s, w_border / s, w_sil / s

def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """mask_a / mask_b 都是 bool 或 0/1 数组"""
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def decode_mask_bool(rle):
    m = mask_util.decode(rle)
    return m.astype(bool) if m.ndim == 2 else m[:, :, 0].astype(bool)


# ======================================================
# 评分函数（NEW，返回 score 和 公式字符串）
# ======================================================
def compute_scores_new(masks_bool, W, H, q_border=0.25):
    """
    Returns:
        scores: List[float] in [0,1]
        formulas: List[str] showing normalized-weight convex combination
    Notes:
        - Area term: piecewise parabola peaked at t_area
        - Silhouette term: v2 (solidity * (1-frag) * (1-hole_ratio))
        - Weights are L1-normalized so final score is in [0,1]
        - Border term uses E = clip(q / d_max, 0, 1)
    """
    img_area = W * H
    cx, cy = W / 2, H / 2

    dist_map = compute_distance_transform(H, W)
    d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

    # normalized weights (convex combination)
    wA, wC, wB, wS = normalize_weights(W_AREA, W_CENTER, W_BORDER, W_SIL)

    scores = []
    formulas = []

    for seg in masks_bool:
        area_px = int(seg.sum())
        if area_px < MIN_AREA_RATIO * img_area:
            scores.append(0.0)
            formulas.append("score = 0.000 (area too small)")
            continue

        # ---- Area term (piecewise parabola) ----
        A_raw = area_px / img_area
        A = area_term_parabola(A_raw, t_area)

        # ---- Center term ----
        ys, xs = np.where(seg)
        mx, my = xs.mean(), ys.mean()
        Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
        C = 1.0 - float(np.clip(Dp, 0.0, 1.0))

        # ---- Border term (quantile distance to border) ----
        q = float(np.quantile(dist_map[seg], q_border))
        E = float(np.clip(q / d_max, 0.0, 1.0))

        # ---- Silhouette term (v2) ----
        Sil = compute_silhouette_score_v2(seg)

        # ---- Final score: convex combination in [0,1] ----
        S = wA * A + wC * C + wB * E + wS * Sil
        S = float(np.clip(S, 0.0, 1.0))

        # ---- Formula string (show raw + normalized) ----
        formula = (
            f"score = "
            f"{wA:.3f}×{A:.3f} (area_parabola; raw={A_raw:.3f}, d={t_area:.3f}) + "
            f"{wC:.3f}×{C:.3f} (center) + "
            f"{wB:.3f}×{E:.3f} (border; q={q_border:.3f}) + "
            f"{wS:.3f}×{Sil:.3f} (sil_v2) "
            f"= {S:.3f}  |  "
            f"rawW=({W_AREA:.2f},{W_CENTER:.2f},{W_BORDER:.2f},{W_SIL:.2f})"
        )

        scores.append(S)
        formulas.append(formula)

    return scores, formulas



# ======================================================
# （可选）初始化 OLD 分数写回 JSON（保持你现有逻辑）
# ======================================================
def write_json(fp: Path, data: dict, backup: bool):
    if backup:
        with open(fp.with_suffix(fp.suffix + ".bak"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compute_scores_original_static(masks_bool, W, H, q_border=0.25):
    """OLD: 固定参数（k=15, t=0.4；权重 0.2/0.2/0.6），仅用于写回旧分数"""
    img_area = W * H
    cx, cy = W / 2, H / 2
    dist_map = compute_distance_transform(H, W)
    d_max = dist_map.max() if dist_map.max() > 0 else 1.0

    scores = []
    for seg in masks_bool:
        area = int(seg.sum())
        if area < MIN_AREA_RATIO * img_area:
            scores.append(0.0)
            continue
        A = area / img_area
        A_sig = 1 / (1 + np.exp(-15.0 * (A - 0.4)))
        ys, xs = np.where(seg)
        mx, my = xs.mean(), ys.mean()
        Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
        C = 1.0 - np.clip(Dp, 0, 1)
        q = float(np.quantile(dist_map[seg], q_border))
        Bp = 1.0 - (q / d_max)
        E = 1.0 - np.clip(Bp, 0, 1)
        S = 0.2 * A_sig + 0.2 * C + 0.6 * E
        scores.append(float(S))
    return scores


# ======================================================
# 正确的匹配率计算
# ======================================================
def compute_match_rates(json_files, score_mode="new"):
    """
    返回 (best_match_rate, in_top3_match_rate)
    - best_match_rate: 算法 Top1 == 人工 rank_1 的比例
    - in_top3_match_rate: 算法 Top1 ∈ {人工 rank_1, rank_2, rank_3} 的比例
    分母仅统计“有人工标注”的样本。
    """
    best_hits = 0
    in_top3_hits = 0
    valid = 0

    for fp in json_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        anns = data.get("annotations", [])
        if not anns:
            continue

        manual = data.get("manual_best_masks", {})
        # 收集人工前3（可能 1~3 个，允许少于 3）
        human_list = []
        for i in (1, 2, 3):
            v = manual.get(f"rank_{i}")
            if isinstance(v, int):
                human_list.append(v)

        if not human_list:
            # 没有人为标注则跳过，不计入分母
            continue

        H, W = data["img_height"], data["img_width"]
        if score_mode == "new":
            masks_bool = [decode_mask_bool(a["segmentation"]) for a in anns]
            scores, _ = compute_scores_new(masks_bool, W, H, q_border=Q_BORDER)
        elif score_mode == "old":
            scores = [a.get("score_custom", 0.0) for a in anns]
        elif score_mode == "sam":
            scores = [a.get("score_sam", 0.0) for a in anns]
        else:
            continue

        # 算法 Top1（单个）
        algo_top1 = int(np.argmax(scores))

        # 人工 rank_1（若不存在就用 None）
        human_best = manual.get("rank_1", None)

        valid += 1
        if human_best is not None and algo_top1 == human_best:
            best_hits += 1
        if algo_top1 in set(human_list):
            in_top3_hits += 1

    if valid == 0:
        return 0.0, 0.0
    return best_hits / valid, in_top3_hits / valid


# ======================================================
# 主逻辑
# ======================================================
if not os.path.isdir(folder):
    st.info("👈 输入有效 JSON 文件夹路径")
    st.stop()

json_files = sorted(Path(folder).glob("*_result.json"))
if not json_files:
    st.warning("❌ 没找到 *_result.json")
    st.stop()

# （可选）同步 Old 分数（仅在你打开开关时执行）
if do_write_json:
    for fp in json_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        anns = data["annotations"]
        H, W = data["img_height"], data["img_width"]
        masks_bool = [decode_mask_bool(a["segmentation"]) for a in anns]
        init_scores = compute_scores_original_static(masks_bool, W, H, q_border=Q_BORDER_INIT)
        best_idx = int(np.argmax(init_scores))
        best_score = init_scores[best_idx]

        changed = False
        for ann, s in zip(anns, init_scores):
            if ann.get("score_custom") != s:
                ann["score_custom"] = float(s)
                changed = True
            if ann.get("score_custom_initial") != s:
                ann["score_custom_initial"] = float(s)
                changed = True
        if data.get("best_mask_index") != best_idx:
            data["best_mask_index"] = best_idx
            changed = True
        if data.get("best_score_custom") != best_score:
            data["best_score_custom"] = float(best_score)
            changed = True
        if changed:
            write_json(fp, data, do_backup)

# ======================================================
# 🔁 Global averages (follow current scoring params)
# 使用当前 slider 下的 compute_scores_new 结果
# ======================================================
human_top_scores = []
nonhuman_scores = []
sam_top_scores = []

for fp in json_files:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    anns = data.get("annotations", [])
    if not anns:
        continue

    manual = data.get("manual_best_masks", {})
    human_top_indices = [
        manual.get(f"rank_{i}") for i in (1, 2, 3)
        if isinstance(manual.get(f"rank_{i}"), int)
    ]

    H, W = data["img_height"], data["img_width"]
    masks_bool = [decode_mask_bool(a["segmentation"]) for a in anns]

    # 使用当前参数的打分函数
    new_scores, _ = compute_scores_new(
        masks_bool, W, H, q_border=Q_BORDER
    )

    # a & b: Human Top3 vs 非 Human Top3（用当前 new_scores）
    for idx, s in enumerate(new_scores):
        if idx in human_top_indices:
            human_top_scores.append(s)
        else:
            nonhuman_scores.append(s)

    # c: SAM Top1 对应的当前打分函数的得分
    sam_scores = [a.get("score_sam", 0.0) for a in anns]
    if sam_scores:
        sam_top_idx = int(np.argmax(sam_scores))  # SAM 用自己的 score_sam 选 Top1
        sam_top_scores.append(new_scores[sam_top_idx])  # 但统计的是当前 new_scores 上的得分

# 三个均值
mean_human_top3 = float(np.mean(human_top_scores)) if human_top_scores else 0.0
mean_nonhuman = float(np.mean(nonhuman_scores)) if nonhuman_scores else 0.0
mean_sam_top = float(np.mean(sam_top_scores)) if sam_top_scores else 0.0

st.markdown("### 🧠 Global Averages (current scoring params)")
cols_static = st.columns(3)
cols_static[0].metric("Human Top3 Avg (new score)", f"{mean_human_top3:.3f}")
cols_static[1].metric("Non-Human Avg (new score)", f"{mean_nonhuman:.3f}")
cols_static[2].metric("SAM Top1 Avg (new score on SAM Top1)", f"{mean_sam_top:.3f}")
st.markdown("---")

# --- 计算六项比例（基于“有人工标注”的样本） ---
new_best, new_in3 = compute_match_rates(json_files, "new")
old_best, old_in3 = compute_match_rates(json_files, "old")
sam_best, sam_in3 = compute_match_rates(json_files, "sam")

st.markdown("### 📊 Performance Overview (only images WITH manual labels)")
cols = st.columns(3)
cols[0].metric("🟩 New: Top1==Human Best", f"{new_best * 100:.1f}%")
cols[0].metric("🟩 New: Top1∈Human Top3", f"{new_in3 * 100:.1f}%")
cols[1].metric("🟦 Old: Top1==Human Best", f"{old_best * 100:.1f}%")
cols[1].metric("🟦 Old: Top1∈Human Top3", f"{old_in3 * 100:.1f}%")
cols[2].metric("🟨 SAM: Top1==Human Best", f"{sam_best * 100:.1f}%")
cols[2].metric("🟨 SAM: Top1∈Human Top3", f"{sam_in3 * 100:.1f}%")

st.markdown("---")

# ======================================================
# IoU 统计：Human rank_1 作为 GT
# - new_score Top1 vs Human Best
# - SAM Top1 vs Human Best
# ======================================================
ious_new_top1 = []
ious_sam_top1 = []
ious_refined_top1 = []

refine_fail_cases = []

for fp in json_files:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    anns = data.get("annotations", [])
    if not anns:
        continue

    manual = data.get("manual_best_masks", {})
    human_best = manual.get("rank_1")
    if not isinstance(human_best, int):
        continue
    if not (0 <= human_best < len(anns)):
        continue

    H, W = data["img_height"], data["img_width"]
    masks_bool = [decode_mask_bool(a["segmentation"]) for a in anns]
    gt_mask = masks_bool[human_best]

    # ---- score-top1 (your method) ----
    new_scores, _ = compute_scores_new(masks_bool, W, H, q_border=Q_BORDER)
    new_top1_idx = int(np.argmax(new_scores))
    new_top1_mask = masks_bool[new_top1_idx]
    iou_new = compute_iou(gt_mask, new_top1_mask)

    # ---- SAM top1 baseline ----
    sam_scores = [a.get("score_sam", 0.0) for a in anns]
    if not sam_scores:
        continue
    sam_top1_idx = int(np.argmax(sam_scores))
    sam_top1_mask = masks_bool[sam_top1_idx]
    iou_sam = compute_iou(gt_mask, sam_top1_mask)

    # ---- NEW: refinement from score-top1 low_res_logits ----
    ann_top1 = anns[new_top1_idx]
    if "low_res_logits" not in ann_top1:
        # 说明你这批 json 还没写 logits，跳过 refined 统计
        continue

    lr = unpack_lowres_logits(ann_top1["low_res_logits"])   # (256,256)
    refined_mask, refined_sam_score = sam2_refine_from_logits(
        data["image_path"],
        ann_top1["bbox"],
        lr
    )
    if refined_mask is None:
        continue

    iou_refined = compute_iou(gt_mask, refined_mask)

    ious_new_top1.append(iou_new)
    ious_sam_top1.append(iou_sam)
    ious_refined_top1.append(iou_refined)

    # 可选：记录 refined 比 new 还差的 case
    if iou_refined < iou_new:
        refine_fail_cases.append({
            "filename": fp.name,
            "iou_new": float(iou_new),
            "iou_refined": float(iou_refined),
            "diff": float(iou_refined - iou_new)
        })


# 只有当两边长度一致时才统计
if ious_new_top1 and len(ious_new_top1) == len(ious_sam_top1):
    arr_new = np.array(ious_new_top1, dtype=float)
    arr_sam = np.array(ious_sam_top1, dtype=float)
    arr_ref = np.array(ious_refined_top1, dtype=float)

    mean_new = float(np.mean(arr_new))
    median_new = float(np.median(arr_new))
    max_new = float(np.max(arr_new))
    min_new = float(np.min(arr_new))

    mean_sam = float(np.mean(arr_sam))
    median_sam = float(np.median(arr_sam))
    max_sam = float(np.max(arr_sam))
    min_sam = float(np.min(arr_sam))

    better_count = int(np.sum(arr_new > arr_sam))
    total_count = len(arr_new)
    better_ratio = better_count / total_count if total_count > 0 else 0.0

    st.markdown("### 📏 IoU Stats (Human rank_1 as GT)")
    st.write(f"Valid images with human rank_1: **{total_count}**")

    cols_iou = st.columns(3)
    with cols_iou[0]:
        st.markdown("**New Score Top1 vs Human Best**")
        st.write(f"- Mean IoU: `{mean_new:.3f}`")
        st.write(f"- Median IoU: `{median_new:.3f}`")
        st.write(f"- Max IoU: `{max_new:.3f}`")
        st.write(f"- Min IoU: `{min_new:.3f}`")

    with cols_iou[1]:
        st.markdown("**SAM Top1 vs Human Best**")
        st.write(f"- Mean IoU: `{mean_sam:.3f}`")
        st.write(f"- Median IoU: `{median_sam:.3f}`")
        st.write(f"- Max IoU: `{max_sam:.3f}`")
        st.write(f"- Min IoU: `{min_sam:.3f}`")
    with cols_iou[2]:
        st.markdown("**Refined (mask-prompt) vs Human Best**")
        st.write(f"- Mean IoU: `{float(np.mean(arr_ref)):.3f}`")
        st.write(f"- Median IoU: `{float(np.median(arr_ref)):.3f}`")
        st.write(f"- Max IoU: `{float(np.max(arr_ref)):.3f}`")
        st.write(f"- Min IoU: `{float(np.min(arr_ref)):.3f}`")
    st.write(
        f"比较同一张图：**New Score Top1 IoU > SAM Top1 IoU** 的比例："
        f"`{better_ratio * 100:.1f}%` ({better_count}/{total_count})"
    )

    # 统计 IoU >= 的比例
    gte_count = int(np.sum(arr_new >= arr_sam))
    gte_ratio = gte_count / total_count if total_count > 0 else 0.0

    st.write(
        f"New Score Top1 IoU ≥ SAM Top1 IoU 的比例："
        f"`{gte_ratio * 100:.1f}%` ({gte_count}/{total_count})"
    )

    # 统计 IoU < 的比例
    gte_count = int(np.sum(arr_new < arr_sam))
    gte_ratio = gte_count / total_count if total_count > 0 else 0.0

    st.write(
        f"New Score Top1 IoU < SAM Top1 IoU 的比例："
        f"`{gte_ratio * 100:.1f}%` ({gte_count}/{total_count})"
    )

    # ---------------------------
    # 去掉两者都完美（IoU_new = IoU_sam = 1）的样本
    # 再统计：New > SAM
    # ---------------------------
    mask_not_both1 = ~((arr_new == 1.0) & (arr_sam == 1.0))
    arr_new_n1 = arr_new[mask_not_both1]
    arr_sam_n1 = arr_sam[mask_not_both1]

    count_n1 = len(arr_new_n1)

    if count_n1 > 0:
        better_n1 = int(np.sum(arr_new_n1 > arr_sam_n1))
        ratio_n1 = better_n1 / count_n1

        st.write(
            f"去除 IoU_new = IoU_sam = 1 的样本后："
            f"New Score Top1 IoU > SAM 的比例："
            f"`{ratio_n1 * 100:.1f}%` ({better_n1}/{count_n1})"
        )

        better_n1 = int(np.sum(arr_new_n1 >= arr_sam_n1))
        ratio_n1 = better_n1 / count_n1

        st.write(
            f"去除 IoU_new = IoU_sam = 1 的样本后："
            f"New Score Top1 IoU >= SAM 的比例："
            f"`{ratio_n1 * 100:.1f}%` ({better_n1}/{count_n1})"
        )
    else:
        st.write("🟧 所有样本都是 IoU=1，无可比较的样本。")
    st.markdown("### 🔬 Statistical Significance Test (Wilcoxon signed-rank)")


    def wilcoxon_report(a, b, title: str):
        diff = a - b
        mask_nonzero = diff != 0
        a_nz = a[mask_nonzero]
        b_nz = b[mask_nonzero]
        N = len(a_nz)

        st.markdown(f"**{title}**")
        if N == 0:
            st.write("All pairs are equal; no non-zero differences for Wilcoxon.")
            return

        stat, p_value = wilcoxon(a_nz, b_nz, alternative="greater")

        mean_W = N * (N + 1) / 4
        var_W = N * (N + 1) * (2 * N + 1) / 24
        z = (stat - mean_W - 0.5) / np.sqrt(var_W)
        effect_r = z / np.sqrt(N)

        st.write(f"- N (diff ≠ 0): `{N}`")
        st.write(f"- W⁺: `{stat:.3f}`")
        st.write(f"- p-value (one-sided, A > B): `{p_value:.2e}`")
        st.write(f"- effect size r: `{effect_r:.3f}`")


    wilcoxon_report(arr_ref, arr_new, "H1: Refined IoU > New-score Top1 IoU")
    wilcoxon_report(arr_ref, arr_sam, "H1: Refined IoU > SAM Top1 IoU")
    # ======================================================
    # 🔍 Fail Cases: SAM IoU > New Score IoU
    # ======================================================
    st.markdown("### 🔍 Fail Cases: SAM IoU > New Score IoU")

    # if not fail_cases:
    #     st.write("🎉 SAM 没有出现比 Score 更好的样本。")
    # else:
    #     # 按差值从大到小排序
    #     fail_cases_sorted = sorted(fail_cases, key=lambda d: d["diff"], reverse=True)
    #
    #     top_k = 30  # 想看多少行就改这里
    #     st.write(f"共发现 {len(fail_cases)} 个 fail cases，展示前 {top_k} 个：")
    #     st.write("")
    #
    #     for case in fail_cases_sorted[:top_k]:
    #         st.write(
    #             f"- **{case['filename']}**  | "
    #             f"IoU_new=`{case['iou_new']:.3f}`  | "
    #             f"IoU_sam=`{case['iou_sam']:.3f}`  | "
    #             f"ΔIoU=`{case['diff']:.3f}`"
    #         )


else:
    st.markdown("### 📏 IoU Stats (Human rank_1 as GT)")
    st.info("没有足够的带 rank_1 的样本来计算 IoU 统计。")

# --- 展示每张图 ---
for fp in json_files:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    anns = data["annotations"]
    if not anns:
        continue

    H, W = data["img_height"], data["img_width"]
    masks_bool = [decode_mask_bool(a["segmentation"]) for a in anns]

    # 新评分 + 公式
    new_scores, new_formulas = compute_scores_new(
        masks_bool, W, H, q_border=Q_BORDER
    )

    # 将 ann, score, formula 绑定在一起，按 new_score 排序取 Top3
    anns_sorted = sorted(
        zip(anns, new_scores, new_formulas),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    manual = data.get("manual_best_masks", {})
    # {mask_id: human_rank}
    human_ranks = {
        manual.get(f"rank_{i}"): i
        for i in range(1, 4)
        if isinstance(manual.get(f"rank_{i}"), int)
    }

    st.markdown(f"### {fp.name}")
    annotated_path = Path(folder) / f"{Path(data['image_path']).stem}_annotated.jpg"
    annotated = Image.open(annotated_path) if annotated_path.exists() else Image.new("RGB", (480, 360), (100, 100, 100))

    small_w = annotated.width // 4
    small_h = annotated.height // 4
    cols_row = st.columns(6)
    cols_row[0].image(annotated.resize((small_w, small_h)), caption="Annotated", use_container_width='always')

    # 先拿 Top3（新分数 + 公式）
    anns_sorted = sorted(
        zip(anns, new_scores, new_formulas),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    # ---- Top1 ----
    ann1, S1, formula1 = anns_sorted[0]
    mask_id1 = ann1["mask_index"]
    human_tag1 = f"Human Top{human_ranks[mask_id1]}" if mask_id1 in human_ranks else "–"

    m1 = mask_util.decode(ann1["segmentation"])
    if m1.ndim == 3: m1 = m1[:, :, 0]
    mask1_img = Image.fromarray((m1 * 255).astype(np.uint8))

    cols_row[1].image(
        mask1_img.resize((small_w, small_h)),
        caption=(f"**Top1: Mask {mask_id1} ({human_tag1})**\n"
                 f"New={S1:.3f} | Old={ann1['score_custom']:.3f} | SAM={ann1['score_sam']:.3f}\n\n{formula1}"),
        use_container_width='always'
    )

    # ---- Refined (from Top1 low_res_logits) ----
    refined_caption = "**Refined**\n(no low_res_logits)"
    if "low_res_logits" in ann1:
        lr = unpack_lowres_logits(ann1["low_res_logits"])
        refined_mask, refined_sam_score = sam2_refine_from_logits(
            data["image_path"],
            ann1["bbox"],
            lr
        )
        if refined_mask is not None:
            ref_img = Image.fromarray((refined_mask.astype(np.uint8) * 255))
            refined_caption = (f"**Refined (Top1 as mask-prompt)**\n"
                               f"SAM(refined)={refined_sam_score:.3f}")

            # 如果这张图有人标 rank_1，则同时显示 IoU(refined)
            human_best = data.get("manual_best_masks", {}).get("rank_1")
            if isinstance(human_best, int) and 0 <= human_best < len(masks_bool):
                gt = masks_bool[human_best]
                iou_ref = compute_iou(gt, refined_mask)
                refined_caption += f"\nIoU(refined vs rank_1)={iou_ref:.3f}"

            cols_row[2].image(ref_img.resize((small_w, small_h)), caption=refined_caption, use_container_width='always')
        else:
            cols_row[2].markdown("_Refine failed_")
    else:
        cols_row[2].markdown("_No logits_")

    # ---- Top2 / Top3 ----
    for out_col, rank in zip([3, 4], [1, 2]):
        if rank < len(anns_sorted):
            ann, S_new, formula = anns_sorted[rank]
            mask_id = ann["mask_index"]
            human_tag = f"Human Top{human_ranks[mask_id]}" if mask_id in human_ranks else "–"

            m = mask_util.decode(ann["segmentation"])
            if m.ndim == 3: m = m[:, :, 0]
            mask_img = Image.fromarray((m * 255).astype(np.uint8))

            cols_row[out_col].image(
                mask_img.resize((small_w, small_h)),
                caption=(f"**Top{rank + 1}: Mask {mask_id} ({human_tag})**\n"
                         f"New={S_new:.3f} | Old={ann['score_custom']:.3f} | SAM={ann['score_sam']:.3f}\n\n{formula}"),
                use_container_width='always'
            )
        else:
            cols_row[out_col].markdown("_No Mask_")

    # ---- SAM Top ----
    best_sam_ann = max(anns, key=lambda a: a.get("score_sam", 0.0))
    mask_id = best_sam_ann["mask_index"]
    human_tag = f"Human Top{human_ranks[mask_id]}" if mask_id in human_ranks else "–"
    m_sam = mask_util.decode(best_sam_ann["segmentation"])
    if m_sam.ndim == 3: m_sam = m_sam[:, :, 0]
    sam_mask_img = Image.fromarray((m_sam * 255).astype(np.uint8))
    cols_row[5].image(
        sam_mask_img.resize((small_w, small_h)),
        caption=(f"**SAM Top — Mask {mask_id} ({human_tag})**\n"
                 f"SAM={best_sam_ann['score_sam']:.3f}"),
        use_container_width='always'
    )

    st.divider()
