import math
from pathlib import Path

import numpy as np
import cv2
import streamlit as st

# ==============================
# Scoring (same as your latest)
# ==============================
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

def compute_terms(seg_bool, W, H, Q_BORDER, t_area):
    img_area = W * H
    cx, cy = W / 2, H / 2

    dist_map = compute_distance_transform(H, W)
    d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

    area_px = int(seg_bool.sum())
    if area_px == 0:
        return dict(A_raw=0.0, A=0.0, C=0.0, B=0.0, Sil=0.0)

    A_raw = area_px / img_area
    A = area_term_parabola(A_raw, t_area)

    ys, xs = np.where(seg_bool)
    mx, my = xs.mean(), ys.mean()
    Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
    C = 1.0 - float(np.clip(Dp, 0.0, 1.0))

    q = float(np.quantile(dist_map[seg_bool], Q_BORDER)) if np.any(seg_bool) else 0.0
    E = float(np.clip(q / d_max, 0.0, 1.0))

    Sil = compute_silhouette_score_v2(seg_bool)
    return dict(A_raw=float(A_raw), A=float(A), C=float(C), B=float(E), Sil=float(Sil))

def compute_scores(seg_bool, W, H, W_AREA, W_CENTER, W_BORDER, W_SIL, Q_BORDER, t_area, MIN_AREA_RATIO):
    img_area = W * H
    area_px = int(seg_bool.sum())

    if area_px < MIN_AREA_RATIO * img_area:
        # hard gate
        return dict(
            gated=True,
            score_mul=0.0,
            score_weighted=0.0,
        )

    terms = compute_terms(seg_bool, W, H, Q_BORDER, t_area)
    A, C, B, Sil = terms["A"], terms["C"], terms["B"], terms["Sil"]

    score_mul = float(np.clip(A * C * B * Sil, 0.0, 1.0))

    wA, wC, wB, wS = normalize_weights(W_AREA, W_CENTER, W_BORDER, W_SIL)
    score_weighted = float(np.clip(wA * A + wC * C + wB * B + wS * Sil, 0.0, 1.0))

    return dict(
        gated=False,
        score_mul=score_mul,
        score_weighted=score_weighted,
    )

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(layout="wide")
st.title("🧪 Synthetic Masks Gallery (A, C, B(E), Sil)")

# Sidebar: folder + params
st.sidebar.header("📁 Data folder")
folder = st.sidebar.text_input(
    "Folder containing .png/.npy",
    value="synthetic_masks_for_score_validation",
    help="Point this to the extracted zip folder (must contain .png).",
)
root = Path(folder)
if not root.exists():
    st.error(f"Folder not found: {root.resolve()}")
    st.stop()

pngs = sorted(root.glob("*.png"))
if not pngs:
    st.error("No .png files found in the folder.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Scoring parameters")
W_AREA = st.sidebar.slider("W_AREA", 0.0, 1.0, 0.2, 0.05)
W_CENTER = st.sidebar.slider("W_CENTER", 0.0, 1.0, 0.2, 0.05)
W_BORDER = st.sidebar.slider("W_BORDER", 0.0, 1.0, 0.6, 0.05)
W_SIL = st.sidebar.slider("W_SIL", 0.0, 1.0, 0.2, 0.05)
Q_BORDER = st.sidebar.slider("Q_BORDER", 0.0, 1.0, 0.25, 0.01)
t_area = st.sidebar.slider("t_area", 0.0, 1.0, 0.4, 0.005)
MIN_AREA_RATIO = st.sidebar.slider("MIN_AREA_RATIO", 0.0, 0.1, 0.01, 0.005)

st.sidebar.markdown("---")
cols_per_row = st.sidebar.slider("Columns per row", 2, 6, 4, 1)
sort_mode = st.sidebar.selectbox("Sort", ["Name (A→Z)", "score_mul (desc)", "score_weighted (desc)"], index=0)
show_weighted = st.sidebar.checkbox("Show weighted score", value=True)

# Header formulas (use st.latex to avoid markdown math rendering issues)
st.subheader("📌 Formula")
st.write("Multiplicative (you requested):")
st.latex(r"\text{score}_{mul} = A \times C \times B \times Sil")
if show_weighted:
    st.write("Weighted (your current system):")
    st.latex(r"\text{score}_{weighted} = w_A A + w_C C + w_B B + w_{Sil} Sil")

# Load all masks + compute stats once
items = []
for p in pngs:
    name = p.stem
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    H, W = img.shape[:2]
    seg = (img > 0)
    terms = compute_terms(seg, W, H, Q_BORDER, t_area)
    scores = compute_scores(seg, W, H, W_AREA, W_CENTER, W_BORDER, W_SIL, Q_BORDER, t_area, MIN_AREA_RATIO)
    area_ratio = float(seg.sum()) / float(W * H)
    items.append({
        "name": name,
        "img": img,
        "H": H, "W": W,
        "area_ratio": area_ratio,
        "terms": terms,
        "scores": scores,
        "png_path": p,
        "npy_path": root / f"{name}.npy",
    })

if not items:
    st.error("Failed to load any images.")
    st.stop()

# Sorting
if sort_mode == "Name (A→Z)":
    items.sort(key=lambda x: x["name"].lower())
elif sort_mode == "score_mul (desc)":
    items.sort(key=lambda x: x["scores"]["score_mul"], reverse=True)
else:
    items.sort(key=lambda x: x["scores"]["score_weighted"], reverse=True)

# Gallery grid
rows = math.ceil(len(items) / cols_per_row)
idx = 0
for _ in range(rows):
    cols = st.columns(cols_per_row, gap="large")
    for c in cols:
        if idx >= len(items):
            break
        it = items[idx]
        idx += 1

        with c:
            st.markdown(f"### {it['name']}")
            st.image(it["img"], clamp=True, use_container_width=True)

            t = it["terms"]
            s = it["scores"]

            # quick badges
            if s["gated"]:
                st.error("GATED (area < MIN_AREA_RATIO)")

            st.caption(f"Area ratio (A_raw): {it['area_ratio']:.6f}")

            # show terms
            st.write(f"**A** = {t['A']:.4f}")
            st.write(f"**C** = {t['C']:.4f}")
            st.write(f"**B(E)** = {t['B']:.4f}")
            st.write(f"**Sil** = {t['Sil']:.4f}")

            # show multiplicative formula with actual numbers
            st.write("**score_mul:**")
            st.latex(
                rf"{t['A']:.4f}\times {t['C']:.4f}\times {t['B']:.4f}\times {t['Sil']:.4f}"
                rf"= {s['score_mul']:.6f}"
            )

            if show_weighted:
                st.write("**score_weighted:**")
                st.write(f"{s['score_weighted']:.6f}")

            # optional info
            if it["npy_path"].exists():
                st.caption(f"NPY: {it['npy_path'].name}")
