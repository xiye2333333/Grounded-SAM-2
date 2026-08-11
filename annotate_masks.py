import os
import json
import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from datetime import datetime
import pycocotools.mask as mask_util

# ==============================
# ⚙️ Streamlit 基本配置
# ==============================
st.set_page_config(layout="wide")
st.title("🖋️ Manual Mask Annotation Tool (v2)")

# === 输入文件夹 ===
folder = st.text_input("📂 JSON Folder", "outputs/AllMasks_v2_score_record")
mask_folder = Path(folder)
json_files = sorted(mask_folder.glob("*_result.json"))

if not json_files:
    st.error("❌ 未找到 *_result.json 文件，请确认路径。")
    st.stop()

# === 当前索引 ===
if "idx" not in st.session_state:
    st.session_state["idx"] = 0
idx = st.session_state["idx"]

# === 当前文件 ===
fp = json_files[idx]
with open(fp, "r", encoding="utf-8") as f:
    data = json.load(f)
anns = data.get("annotations", [])
h, w = data.get("img_height", 720), data.get("img_width", 1280)

# === 加载原图与 Annotated ===
img_id = Path(data["image_path"]).stem
orig_path = Path("outputs/AllMasks_v2_score") / f"{img_id}.jpg.png"
annot_path = Path(folder) / f"{img_id}_annotated.jpg"

img_orig = Image.open(orig_path) if orig_path.exists() else Image.new("RGB", (w, h), (80, 80, 80))
img_annot = Image.open(annot_path) if annot_path.exists() else Image.new("RGB", (w, h), (60, 60, 60))

# === 顶部信息栏 ===
st.markdown(f"### 📄 {fp.name}  ({idx+1}/{len(json_files)})")

# === 左右布局 ===
col1, col2 = st.columns(2)
col1.image(img_orig, caption="Original Image", width=400)
col2.image(img_annot, caption="SAM Annotated", width=400)

st.markdown("#### 🩶 Candidate Masks")

# === 解码所有 mask 并展示 ===
mask_ids = [a["mask_index"] for a in anns]
cols = st.columns(6)
for i, ann in enumerate(anns):
    m = mask_util.decode(ann["segmentation"])
    if m.ndim == 3:
        m = m[:, :, 0]
    img_mask = Image.fromarray((m * 255).astype(np.uint8))
    col = cols[i % 6]
    col.image(img_mask.resize((150, 150)), caption=f"Mask {ann['mask_index']}", width=150)
    if (i + 1) % 6 == 0 and (i + 1) < len(anns):
        cols = st.columns(6)

st.markdown("---")

# === 可选 Mask ID 列表 ===
mask_ids_with_none = ["None"] + [str(i) for i in mask_ids]

rank_1 = st.selectbox("🥇 Best Mask (Rank 1)", mask_ids_with_none, index=0)
rank_2 = st.selectbox("🥈 Second Best (Rank 2)", mask_ids_with_none, index=0)
rank_3 = st.selectbox("🥉 Third Best (Rank 3)", mask_ids_with_none, index=0)

# === 解析选择 ===
def parse_rank(val):
    try:
        return int(val) if val != "None" else None
    except Exception:
        return None

# === 保存选择 ===
if st.button("✅ Confirm Selection"):
    data["manual_best_masks"] = {
        "rank_1": parse_rank(rank_1),
        "rank_2": parse_rank(rank_2),
        "rank_3": parse_rank(rank_3),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "annotated_by": "Human"
    }

    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    st.success(f"✅ 已保存 {fp.name} 的人工标注结果！")

# === 下一张按钮 ===
col_left, col_right = st.columns([1, 5])
with col_left:
    if st.button("➡️ Next Image"):
        if idx < len(json_files) - 1:
            st.session_state["idx"] = idx + 1
            st.rerun()
        else:
            st.success("🎉 所有图片标注完毕！")
            st.session_state["idx"] = 0
with col_right:
    st.progress((idx + 1) / len(json_files))

st.markdown("---")
st.caption("💡 你可以选择 1～3 个 mask；未选的 rank 会写入 JSON 为 null。")
