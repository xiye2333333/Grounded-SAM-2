import os
import re
import json
import math
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


# -----------------------------
# Config (edit these paths)
# -----------------------------
MY_GT_DIR = r"D:\uwb thesis\code\Grounded-SAM-2\GT_masks"
VOC_SUMMARY_DIR = r"D:\uwb thesis\code\Grounded-SAM-2\voc2012_signature_out"  # where summary_overall_trainval.json lives
VOC_SPLIT = "trainval"

OUTPUT_DIR = r"D:\uwb thesis\code\Grounded-SAM-2\my_dataset_signature_out"  # new output dir
MIN_COMPONENT_AREA_PX = 20  # ignore tiny fragments


# -----------------------------
# Fixed bin settings (match VOC for comparability)
# -----------------------------
SIZE_BINS = 100
CONC_BINS = 50
DC_BINS = 50

SIZE_RANGE = (0.0, 1.0)
CONC_RANGE = (0.0, 1.0)
DC_MAX = float(math.sqrt(0.5**2 + 0.5**2))  # 0.70710678
DC_RANGE = (0.0, DC_MAX)


# -----------------------------
# Helpers
# -----------------------------
def safe_load_json(p: str) -> dict:
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing file: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def fixed_hist(values: List[float], bins: int, vmin: float, vmax: float) -> Dict:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"hist": [0] * bins, "edges": np.linspace(vmin, vmax, bins + 1).tolist()}
    v = np.clip(v, vmin, vmax)
    hist, edges = np.histogram(v, bins=bins, range=(vmin, vmax))
    return {"hist": hist.tolist(), "edges": edges.tolist()}


def bin_centers(edges: List[float]) -> np.ndarray:
    e = np.asarray(edges, dtype=np.float64)
    return 0.5 * (e[:-1] + e[1:])


def normalize_prob(hist: List[int]) -> np.ndarray:
    h = np.asarray(hist, dtype=np.float64)
    s = float(h.sum())
    if s <= 0:
        return np.zeros_like(h)
    return h / s  # 0..1


def concavity_from_component(comp_mask01: np.ndarray) -> float:
    """
    comp_mask01: uint8 {0,1} mask for a single component.
    concavity = 1 - A / A_hull
    """
    A = int(comp_mask01.sum())
    if A < 3:
        return 0.0

    m255 = (comp_mask01.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return 0.0

    hull = cv2.convexHull(cnt)
    Ahull = float(cv2.contourArea(hull))
    if Ahull <= 1e-6:
        Ahull = float(A)

    ratio = float(A) / float(Ahull)
    ratio = max(0.0, min(1.0, ratio))
    return float(1.0 - ratio)


def compute_component_features(mask01: np.ndarray) -> Tuple[List[float], List[float], List[float]]:
    """
    mask01: uint8 {0,1} full image mask (binary).
    Returns lists of s (relative size), concavity, dc (center distance) per connected component.
    """
    H, W = mask01.shape[:2]
    img_area = float(H * W)
    m255 = (mask01.astype(np.uint8) * 255)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m255, connectivity=8)

    s_list, c_list, dc_list = [], [], []
    for lab in range(1, num_labels):  # skip background 0
        area_px = int(stats[lab, cv2.CC_STAT_AREA])
        if area_px < MIN_COMPONENT_AREA_PX:
            continue

        # size
        s = float(area_px) / img_area

        # center distance
        cx, cy = centroids[lab]  # x,y in pixels
        xcn = float(cx) / float(W)
        ycn = float(cy) / float(H)
        dc = math.sqrt((xcn - 0.5) ** 2 + (ycn - 0.5) ** 2)

        # concavity
        comp = (labels == lab).astype(np.uint8)  # 0/1
        conc = concavity_from_component(comp)

        s_list.append(s)
        c_list.append(conc)
        dc_list.append(dc)

    return s_list, c_list, dc_list


# -----------------------------
# Load my dataset GTs
# -----------------------------
def collect_my_dataset_features(gt_dir: str) -> Dict:
    """
    Scan img_????_gt.json under gt_dir.
    For each JSON, read gt_mask_png and compute features over connected components.
    Returns overall summary with fixed hist fields (size_linear_hist, concavity_hist, dc_hist).
    """
    pattern = re.compile(r"^img_\d{4}_gt\.json$", re.IGNORECASE)
    json_paths = [
        os.path.join(gt_dir, fn)
        for fn in os.listdir(gt_dir)
        if pattern.match(fn)
    ]
    json_paths.sort()

    if not json_paths:
        raise RuntimeError(f"No GT json found under {gt_dir} with pattern img_????_gt.json")

    all_s, all_c, all_dc = [], [], []
    n_images = 0
    n_components = 0

    for jp in tqdm(json_paths, desc="Reading my GT json"):
        meta = safe_load_json(jp)
        mask_png = meta.get("gt_mask_png", None)
        if not mask_png or (not os.path.isfile(mask_png)):
            # If png missing, you could implement RLE decode here; for now, fail loudly.
            raise FileNotFoundError(f"gt_mask_png missing or not found for {jp}: {mask_png}")

        m = np.array(Image.open(mask_png))
        if m.ndim == 3:
            # if RGB, treat non-zero as foreground
            mask01 = (m.sum(axis=2) > 0).astype(np.uint8)
        else:
            # if grayscale, treat >0 as fg (VOC uses class ids; yours likely 0/255)
            mask01 = (m > 0).astype(np.uint8)

        s_list, c_list, dc_list = compute_component_features(mask01)
        if len(s_list) == 0:
            continue

        all_s.extend(s_list)
        all_c.extend(c_list)
        all_dc.extend(dc_list)
        n_images += 1
        n_components += len(s_list)

    summary = {
        "name": "MY_DATASET",
        "n_images_used": int(n_images),
        "n_components": int(n_components),
        "size_linear_hist": fixed_hist(all_s, bins=SIZE_BINS, vmin=SIZE_RANGE[0], vmax=SIZE_RANGE[1]),
        "concavity_hist": fixed_hist(all_c, bins=CONC_BINS, vmin=CONC_RANGE[0], vmax=CONC_RANGE[1]),
        "dc_hist": fixed_hist(all_dc, bins=DC_BINS, vmin=DC_RANGE[0], vmax=DC_RANGE[1]),
    }
    return summary


# -----------------------------
# Load VOC overall summary (must have fixed hist fields)
# -----------------------------
def load_voc_overall_summary(voc_summary_dir: str, split: str) -> Dict:
    p = os.path.join(voc_summary_dir, f"summary_overall_{split}.json")
    voc = safe_load_json(p)

    # Require fixed-edge fields produced by the "updated summarize()"
    required = ["size_linear_hist", "concavity_hist", "dc_hist"]
    missing = [k for k in required if k not in voc]
    if missing:
        raise RuntimeError(
            f"VOC overall summary missing fields {missing}.\n"
            f"Please regenerate VOC summaries using the updated analyze_voc() that outputs:\n"
            f"  - size_linear_hist (fixed bins [0,1])\n"
            f"  - concavity_hist (fixed bins [0,1])\n"
            f"  - dc_hist (fixed bins [0,0.7071])\n"
        )

    voc["name"] = f"VOC2012 ({split})"
    return voc


# -----------------------------
# Plotting (Fig6-like style, NO SHOW)
# -----------------------------
def plot_signature_panel(series_size, series_conc, series_dc, out_png: str, title: str = None,
                         legend_outside: bool = True):
    """
    series_*: list of dict {label, hist, edges}
    """
    fig = plt.figure(figsize=(14, 3.6), dpi=220)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    def plot_one(ax, series, xlabel, xlim):
        for item in series:
            x = bin_centers(item["edges"])
            y = normalize_prob(item["hist"])  # 0..1
            ax.plot(x, y, linewidth=1.2, label=item["label"])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability")
        ax.set_xlim(*xlim)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    plot_one(ax1, series_size, "Relative segmentation mask size", (0.0, 1.0))
    plot_one(ax2, series_conc, "Concavity", (0.0, 1.0))
    plot_one(ax3, series_dc, r"Center distance $d_c$", (0.0, DC_MAX))

    if title:
        fig.suptitle(title, y=1.02)

    # Legend handling
    handles, labels = ax1.get_legend_handles_labels()
    if legend_outside:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5),
                   frameon=False, fontsize=9)
        fig.tight_layout(rect=[0, 0, 0.86, 1])
    else:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.92])

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1) My dataset overall summary + save json
    my_summary = collect_my_dataset_features(MY_GT_DIR)
    my_json_out = os.path.join(OUTPUT_DIR, "summary_overall_my_dataset.json")
    with open(my_json_out, "w", encoding="utf-8") as f:
        json.dump(my_summary, f, indent=2)
    print(f"[DONE] Wrote my dataset summary: {my_json_out}")

    # 2) Plot my dataset alone
    plot_signature_panel(
        series_size=[{"label": "MY_DATASET", **my_summary["size_linear_hist"]}],
        series_conc=[{"label": "MY_DATASET", **my_summary["concavity_hist"]}],
        series_dc=[{"label": "MY_DATASET", **my_summary["dc_hist"]}],
        out_png=os.path.join(plots_dir, "my_dataset_overall_signature.png"),
        title=f"MY_DATASET overall (n_images={my_summary['n_images_used']}, n_components={my_summary['n_components']})",
        legend_outside=False,
    )
    print(f"[DONE] Saved: {os.path.join(plots_dir, 'my_dataset_overall_signature.png')}")

    # 3) Load VOC overall summary (must contain fixed bins)
    voc = load_voc_overall_summary(VOC_SUMMARY_DIR, VOC_SPLIT)

    # 4) Plot overlay: MY vs VOC (overall only)
    plot_signature_panel(
        series_size=[
            {"label": voc["name"], **voc["size_linear_hist"]},
            {"label": "MY_DATASET", **my_summary["size_linear_hist"]},
        ],
        series_conc=[
            {"label": voc["name"], **voc["concavity_hist"]},
            {"label": "MY_DATASET", **my_summary["concavity_hist"]},
        ],
        series_dc=[
            {"label": voc["name"], **voc["dc_hist"]},
            {"label": "MY_DATASET", **my_summary["dc_hist"]},
        ],
        out_png=os.path.join(plots_dir, "overlay_my_vs_voc_overall.png"),
        title="Overall distribution comparison",
        legend_outside=True,
    )
    print(f"[DONE] Saved: {os.path.join(plots_dir, 'overlay_my_vs_voc_overall.png')}")
    print(f"[DONE] All plots saved under: {plots_dir}")


if __name__ == "__main__":
    main()
