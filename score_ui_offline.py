import numpy as np
from pathlib import Path

import streamlit as st

import eval_SAM_with_Score as bench  # 你原来的模块，内含 compute_scores_new / compute_iou


# ==========================
# 基本设置
# ==========================

PRECOMPUTED_ROOT = Path("precomputed_masks")  # npz 根目录

st.set_page_config(layout="wide")
st.title("Score Param Playground (Offline, NPZ)")

# 选择类别 ID
class_id = st.sidebar.number_input("VOC class id", min_value=1, max_value=20, value=7, step=1)

# npz 路径
class_dir = PRECOMPUTED_ROOT / f"class_{class_id}"
st.sidebar.write(f"NPZ dir: `{class_dir}`")

# ==========================
# 参数控制（手动调）
# ==========================

st.sidebar.markdown("### Score Parameters")

# 面积过滤比例（0-1）
bench.MIN_AREA_RATIO = st.sidebar.slider("MIN_AREA_RATIO", 0.0, 1.0, float(getattr(bench, "MIN_AREA_RATIO", 0.1)), 0.01)

# Sigmoid 参数（0-2 和 0-1）
bench.k_area = st.sidebar.slider("k_area (0-2)", 0.0, 30.0, float(getattr(bench, "k_area", 1.0)), 0.01)
bench.t_area = st.sidebar.slider("t_area (0-1)", 0.0, 1.0, float(getattr(bench, "t_area", 0.3)), 0.01)

# 四个权重（0-2，不做归一化）
bench.W_AREA = st.sidebar.slider("W_AREA",   0.0, 2.0, float(getattr(bench, "W_AREA",   0.5)), 0.01)
bench.W_CENTER = st.sidebar.slider("W_CENTER", 0.0, 2.0, float(getattr(bench, "W_CENTER", 0.5)), 0.01)
bench.W_BORDER = st.sidebar.slider("W_BORDER", 0.0, 2.0, float(getattr(bench, "W_BORDER", 0.5)), 0.01)
bench.W_SIL = st.sidebar.slider("W_SIL",    0.0, 2.0, float(getattr(bench, "W_SIL",    0.5)), 0.01)

bench.sigmoid_isOn = st.sidebar.checkbox("Use sigmoid on area term", value=getattr(bench, "sigmoid_isOn", True))

# q_border 固定一个 slider（可选）
q_border = st.sidebar.slider("q_border", 0.0, 1.0, 0.15, 0.01)


# ==========================
# 计算 mean IoU
# ==========================

st.markdown(f"### Class {class_id} — Mean IoU (Offline NPZ)")

npz_files = sorted(class_dir.glob("*.npz"))

if not npz_files:
    st.error(f"No npz files found in `{class_dir}`. 请先用 precompute_sam_masks.py 生成。")
    st.stop()

ious_sam = []
ious_ours = []
num_images = 0

for npz_path in npz_files:
    data = np.load(npz_path, allow_pickle=True)

    gt_mask = data["gt_mask"].astype(bool)       # (H, W)
    masks = data["masks"].astype(bool)           # (K, H, W)
    sam_confs = data["sam_confs"].astype(float)  # (K,)

    if masks.ndim != 3 or masks.shape[0] == 0:
        continue

    K, H, W = masks.shape

    # --- SAM: 用 sam_conf 选 Top1 ---
    sam_best_idx = int(np.argmax(sam_confs))
    pred_sam = masks[sam_best_idx]
    iou_sam = bench.compute_iou(pred_sam, gt_mask)

    # --- Ours: 用当前 score 选 Top1 ---
    masks_list = [masks[i] for i in range(K)]
    scores, _ = bench.compute_scores_new(
        masks_list, W=W, H=H, q_border=q_border, sigmoid_isOn=bench.sigmoid_isOn
    )
    scores = np.array(scores)
    ours_best_idx = int(np.argmax(scores))
    pred_ours = masks[ours_best_idx]
    iou_ours = bench.compute_iou(pred_ours, gt_mask)

    ious_sam.append(iou_sam)
    ious_ours.append(iou_ours)
    num_images += 1

if num_images == 0:
    st.error("所有 npz 文件中都没有有效的 masks。")
    st.stop()

sam_mean_iou = float(np.mean(ious_sam))
ours_mean_iou = float(np.mean(ious_ours))

col1, col2, col3 = st.columns(3)
col1.metric("Num Images", num_images)
col2.metric("SAM mean IoU", f"{sam_mean_iou:.4f}")
col3.metric("Ours mean IoU", f"{ours_mean_iou:.4f}")
