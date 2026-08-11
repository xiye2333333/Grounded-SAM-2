import os
import json
from pathlib import Path
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import pycocotools.mask as mask_util

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
    "outputs/AllMasks_v2_score_record"
]
folder = st.sidebar.selectbox("📂 Select JSON Folder", AVAILABLE_FOLDERS, index=0)
st.sidebar.markdown("---")

# ✅ 权重调节栏
W_AREA = st.sidebar.slider("W_AREA", 0.0, 1.0, 0.2, 0.05)
W_CENTER = st.sidebar.slider("W_CENTER", 0.0, 1.0, 0.2, 0.05)
W_BORDER = st.sidebar.slider("W_BORDER", 0.0, 1.0, 0.6, 0.05)
W_SIL = st.sidebar.slider("W_SILHOUETTE", 0.0, 1.0, 0.2, 0.05)
Q_BORDER = st.sidebar.slider("Q_BORDER", 0.0, 1.0, 0.25, 0.01)

t_area = st.sidebar.slider("🎯 Area target ratio (t)", 0.0, 1.0, 0.4, 0.005)
k_area = st.sidebar.slider("⚡ Area steepness (k)", 1.0, 30.0, 15.0, 1.0)
# sigmoid_isOn = st.sidebar.checkbox("Use Sigmoid for Area Term", value=True)

st.sidebar.markdown("---")
do_write_json = st.sidebar.toggle("写回 JSON: Old=初始分数", value=True)
do_backup = st.sidebar.toggle("写入前 *.bak 备份", value=True)

st.title("🔥 Adjustable Scoring UI ┃ With Human Match Metrics + SAM Visualization")

Q_BORDER_INIT = 0.25
MIN_AREA_RATIO = 0.01


# ======================================================
# 工具函数
# ======================================================
def compute_distance_transform(h, w):
    border_mask = np.zeros((h, w), np.uint8)
    border_mask[1:-1, 1:-1] = 1
    return cv2.distanceTransform(border_mask, distanceType=cv2.DIST_L2, maskSize=5)


def compute_silhouette_score(mask: np.ndarray) -> float:
    mask_u8 = (mask.astype(np.uint8) > 0).astype(np.uint8)

    # 1) 紧致度（圆形度）
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    P = float(cv2.arcLength(largest, True))
    A = float(cv2.contourArea(largest))
    if A <= 0 or P <= 0:
        return 0.0
    compactness = float(np.clip((4.0 * np.pi * A) / (P * P), 0.0, 1.0))

    # 2) 连通分量破碎度
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        total = float(np.sum(areas))
        if total <= 0:
            fragmentation = 0.0
        else:
            largest_area = float(np.max(areas))
            fragmentation = 1.0 - (largest_area / total)
    else:
        fragmentation = 0.0
    fragmentation = float(np.clip(fragmentation, 0.0, 1.0))

    # 3) 闭运算检测缝隙
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    closed_sum = float(closed.sum())
    if closed_sum <= 0:
        gap_ratio = 0.0
    else:
        gap_ratio = 1.0 - (float(mask_u8.sum()) / closed_sum)
    gap_ratio = float(np.clip(gap_ratio, 0.0, 1.0))

    sil = compactness * (1.0 - fragmentation) * (1.0 - gap_ratio)
    return float(np.clip(sil, 0.0, 1.0))


def compute_silhouette_score_v2(mask: np.ndarray) -> float:
    """
    Improved silhouette score based on:
    - solidity (area / convex hull area)
    - fragmentation (ignoring tiny components)
    - hole penalty (holes area / mask area), computed robustly via floodfill on background

    Args:
        mask (np.ndarray): bool or uint8 mask of shape (H, W)

    Returns:
        float: silhouette score in [0,1]
    """
    # ---- normalize to {0,1} uint8 ----
    mask_u8 = (mask.astype(np.uint8) > 0).astype(np.uint8)

    # ===== 1) Solidity (area / convex hull area) =====
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

    # ===== 2) Fragmentation (ignore tiny components) =====
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    if num_labels <= 1:
        fragmentation = 0.0
    else:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)  # drop background
        total_area = float(np.sum(areas))
        if total_area <= 0:
            fragmentation = 0.0
        else:
            thr = 0.01 * total_area  # ignore blobs < 1% of total
            large_areas = areas[areas >= thr]

            if large_areas.size <= 1:
                fragmentation = 0.0
            else:
                largest_area = float(np.max(large_areas))
                fragmentation = 1.0 - (largest_area / float(np.sum(large_areas)))

    fragmentation = float(np.clip(fragmentation, 0.0, 1.0))

    # ===== 3) Hole penalty (robust floodfill on background) =====
    # background = 1 - mask; floodfill from border to mark "outside background"
    # remaining background pixels (still 1) are holes inside the mask.
    bg = (1 - mask_u8).astype(np.uint8)  # 1 for background, 0 for foreground

    # Pad to guarantee (0,0) is outside background even if mask touches border
    bg_pad = np.pad(bg, pad_width=1, mode="constant", constant_values=1)

    # floodFill requires a mask that's 2 pixels larger than the image
    h, w = bg_pad.shape
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    bg_ff = bg_pad.copy()
    cv2.floodFill(bg_ff, ff_mask, (0, 0), 0)  # fill outside background with 0

    # holes are remaining background pixels = 1, excluding padding border
    holes_map = (bg_ff[1:-1, 1:-1] == 1)
    holes_area = float(np.count_nonzero(holes_map))

    hole_ratio = 0.0 if area <= 0 else float(np.clip(holes_area / area, 0.0, 1.0))

    # ===== Final score =====
    sil = solidity * (1.0 - fragmentation) * (1.0 - hole_ratio)
    return float(np.clip(sil, 0.0, 1.0))
def decode_mask_bool(rle):
    m = mask_util.decode(rle)
    return m.astype(bool) if m.ndim == 2 else m[:, :, 0].astype(bool)



EPS = 1e-8

def area_term_parabola(x: float, d: float, eps: float = 1e-6) -> float:
    d = float(np.clip(d, eps, 1.0))
    x = float(np.clip(x, 0.0, 1.0))
    val = 1.0 - ((x - d) / d) ** 2
    return float(np.clip(val, 0.0, 1.0))

def normalize_weights(w_area, w_center, w_border, w_sil, eps: float = 1e-8):
    s = float(w_area + w_center + w_border + w_sil)
    if s <= eps:
        # 全 0 时给一个默认均匀权重，避免所有分数都变 0
        return 0.25, 0.25, 0.25, 0.25
    return w_area / s, w_center / s, w_border / s, w_sil / s
# ======================================================
# 评分函数（NEW）
# ======================================================

def compute_scores_new(masks_bool, W, H, q_border=0.25):
    img_area = W * H
    cx, cy = W / 2, H / 2

    dist_map = compute_distance_transform(H, W)
    d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

    # ✅ 权重归一化（保证总分 ∈ [0,1]）
    wA, wC, wB, wS = normalize_weights(W_AREA, W_CENTER, W_BORDER, W_SIL)

    scores = []
    for seg in masks_bool:
        area = int(seg.sum())
        if area < MIN_AREA_RATIO * img_area:
            scores.append(0.0)
            continue

        # ---- Area term: parabola peak at t_area ----
        A_raw = area / img_area
        A = area_term_parabola(A_raw, t_area)

        # ---- Center term ----
        ys, xs = np.where(seg)
        mx, my = xs.mean(), ys.mean()
        Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
        C = 1.0 - float(np.clip(Dp, 0.0, 1.0))

        # ---- Border term ----
        q = float(np.quantile(dist_map[seg], q_border))
        E = float(np.clip(q / d_max, 0.0, 1.0))

        # ---- Silhouette term (v2) ----
        Sil = compute_silhouette_score_v2(seg)

        # ---- Final normalized score in [0,1] ----
        S = wA * A + wC * C + wB * E + wS * Sil
        scores.append(float(np.clip(S, 0.0, 1.0)))

    return scores
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
# 正确的匹配率计算（关键修复）
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
            scores = compute_scores_new(
                masks_bool, W, H,
                q_border=Q_BORDER,
            )
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

# --- 计算六项比例（基于“有人工标注”的样本）
new_best, new_in3 = compute_match_rates(json_files, "new")
old_best, old_in3 = compute_match_rates(json_files, "old")
sam_best, sam_in3 = compute_match_rates(json_files, "sam")

st.markdown("### 📊 Performance Overview (only images WITH manual labels)")
cols = st.columns(3)
cols[0].metric("🟩 New: Top1==Human Best", f"{new_best*100:.1f}%")
cols[0].metric("🟩 New: Top1∈Human Top3", f"{new_in3*100:.1f}%")
cols[1].metric("🟦 Old: Top1==Human Best", f"{old_best*100:.1f}%")
cols[1].metric("🟦 Old: Top1∈Human Top3", f"{old_in3*100:.1f}%")
cols[2].metric("🟨 SAM: Top1==Human Best", f"{sam_best*100:.1f}%")
cols[2].metric("🟨 SAM: Top1∈Human Top3", f"{sam_in3*100:.1f}%")

st.markdown("---")

# --- 展示每张图 ---
for fp in json_files:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    anns = data["annotations"]
    if not anns:
        continue

    H, W = data["img_height"], data["img_width"]
    masks_bool = [decode_mask_bool(a["segmentation"]) for a in anns]
    new_scores = compute_scores_new(masks_bool, W, H, q_border=Q_BORDER)
    anns_sorted = sorted(zip(anns, new_scores), key=lambda x: x[1], reverse=True)[:3]

    manual = data.get("manual_best_masks", {})
    # {mask_id: human_rank}
    human_ranks = {manual.get(f"rank_{i}"): i for i in range(1, 4) if isinstance(manual.get(f"rank_{i}"), int)}

    st.markdown(f"### {fp.name}")
    annotated_path = Path(folder) / f"{Path(data['image_path']).stem}_annotated.jpg"
    annotated = Image.open(annotated_path) if annotated_path.exists() else Image.new("RGB", (480, 360), (100, 100, 100))

    small_w = annotated.width // 4
    small_h = annotated.height // 4
    cols = st.columns(5)
    cols[0].image(annotated.resize((small_w, small_h)), caption="Annotated", width='stretch')

    # === Top1~Top3（新分数） ===
    for col_idx, rank in zip(range(1, 4), range(3)):
        if rank < len(anns_sorted):
            ann, S_new = anns_sorted[rank]
            mask_id = ann["mask_index"]
            human_tag = f"Human Top{human_ranks[mask_id]}" if mask_id in human_ranks else "–"

            m = mask_util.decode(ann["segmentation"])
            if m.ndim == 3: m = m[:, :, 0]
            mask_img = Image.fromarray((m * 255).astype(np.uint8))

            cols[col_idx].image(
                mask_img.resize((small_w, small_h)),
                caption=(f"**Top{rank + 1}: Mask {mask_id} ({human_tag})**\n"
                         f"New={S_new:.3f} | Old={ann['score_custom']:.3f} | SAM={ann['score_sam']:.3f}")
            )
        else:
            cols[col_idx].markdown("_No Mask_")

    # === SAM Top（并注记其在人类标注中的名次） ===
    best_sam_ann = max(anns, key=lambda a: a.get("score_sam", 0.0))
    mask_id = best_sam_ann["mask_index"]
    human_tag = f"Human Top{human_ranks[mask_id]}" if mask_id in human_ranks else "–"

    m_sam = mask_util.decode(best_sam_ann["segmentation"])
    if m_sam.ndim == 3: m_sam = m_sam[:, :, 0]
    sam_mask_img = Image.fromarray((m_sam * 255).astype(np.uint8))
    cols[4].image(
        sam_mask_img.resize((small_w, small_h)),
        caption=(f"**SAM Top — Mask {mask_id} ({human_tag})**\n"
                 f"SAM={best_sam_ann['score_sam']:.3f}")
    )

    st.divider()
