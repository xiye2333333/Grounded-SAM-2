import os
import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import cv2

# ----------------------------
# VOC2012 class mapping
# ----------------------------
VOC_CLASSES = [
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


# 255 is ignore/void in VOC segmentation masks


def find_voc2012_root(voc_root: str) -> str:
    """
    Accepts either:
      - .../VOC2012_train_val/VOC2012_train_val  (user path)
      - .../VOCdevkit/VOC2012
      - .../VOC2012
    Returns a path that directly contains JPEGImages/ SegmentationClass/ ImageSets/
    """
    candidates = [
        voc_root,
        os.path.join(voc_root, "VOC2012"),
        os.path.join(voc_root, "VOCdevkit", "VOC2012"),
        os.path.join(voc_root, "VOCdevkit"),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "JPEGImages")) and \
                os.path.isdir(os.path.join(c, "SegmentationClass")) and \
                os.path.isdir(os.path.join(c, "ImageSets")):
            return c
    # try one more common nesting
    for c in candidates:
        cc = os.path.join(c, "VOC2012")
        if os.path.isdir(os.path.join(cc, "JPEGImages")) and \
                os.path.isdir(os.path.join(cc, "SegmentationClass")) and \
                os.path.isdir(os.path.join(cc, "ImageSets")):
            return cc
    raise FileNotFoundError(
        f"Could not locate VOC2012 folders under: {voc_root}\n"
        f"Expected JPEGImages/, SegmentationClass/, ImageSets/ somewhere inside."
    )


def load_split_ids(voc2012_root: str, split: str = "trainval"):
    split_file = os.path.join(voc2012_root, "ImageSets", "Segmentation", f"{split}.txt")
    if not os.path.isfile(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f.readlines() if line.strip()]
    return ids


def compute_centroid(binary_mask: np.ndarray):
    # binary_mask: uint8 {0,1} or {0,255}
    ys, xs = np.nonzero(binary_mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def concavity_from_component(comp_mask: np.ndarray):
    """
    comp_mask: uint8 {0,1} mask for a single connected component.
    Returns concavity c = 1 - A/A_hull, and (A, A_hull).
    Uses contour -> convex hull -> contourArea for A_hull.
    """
    A = int(comp_mask.sum())
    if A < 3:
        # too small to define hull area; treat as convex
        return 0.0, A, max(A, 1)

    # OpenCV expects 0/255
    m255 = (comp_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, A, max(A, 1)

    # Usually there is exactly one contour for one component, but be safe:
    # merge by taking the largest contour
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return 0.0, A, max(A, 1)

    hull = cv2.convexHull(cnt)
    Ahull = float(cv2.contourArea(hull))
    # If hull area degenerates (e.g., line-like), fall back
    if Ahull <= 1e-6:
        Ahull = float(A)

    ratio = float(A) / float(Ahull)
    ratio = min(max(ratio, 0.0), 1.0)
    c = 1.0 - ratio
    return float(c), A, float(Ahull)


def log_bins(values, bins=20):
    """
    Log bins for values in (0, 1], typical for relative area.
    We'll clip min to 1e-8 for stability.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None

    v = np.clip(v, 1e-8, 1.0)
    edges = np.logspace(np.log10(1e-8), 0, bins + 1)  # 1e-8 to 1
    hist, edges = np.histogram(v, bins=edges)
    return hist.tolist(), edges.tolist()


def percentiles(values, ps=(10, 25, 50, 75, 90)):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"p{p}": None for p in ps}
    out = np.percentile(v, ps)
    return {f"p{p}": float(x) for p, x in zip(ps, out)}


def analyze_voc(
        voc_root: str,
        split: str = "trainval",
        min_component_area_px: int = 20,
        include_background: bool = False,
        out_dir: str = "voc2012_signature_out",
):
    voc2012_root = find_voc2012_root(voc_root)
    print(f"[INFO] Using VOC2012 root: {voc2012_root}")

    ids = load_split_ids(voc2012_root, split=split)
    print(f"[INFO] Loaded split '{split}' with {len(ids)} images")

    seg_dir = os.path.join(voc2012_root, "SegmentationClass")

    os.makedirs(out_dir, exist_ok=True)

    # Collect per-class samples as lists of (s, concavity, dc)
    by_class = defaultdict(lambda: {"s": [], "c": [], "dc": [], "count": 0})
    overall = {"s": [], "c": [], "dc": [], "count": 0}

    classes_to_iter = list(range(len(VOC_CLASSES)))
    if not include_background:
        classes_to_iter = [k for k in classes_to_iter if k != 0]

    for img_id in tqdm(ids, desc="Processing masks"):
        seg_path = os.path.join(seg_dir, f"{img_id}.png")
        if not os.path.isfile(seg_path):
            continue

        seg = np.array(Image.open(seg_path), dtype=np.uint8)
        H, W = seg.shape[:2]
        img_area = float(H * W)

        # ignore void (255)
        valid = (seg != 255)

        for cls_id in classes_to_iter:
            cls_mask = (seg == cls_id) & valid
            if not cls_mask.any():
                continue

            # connected components
            # OpenCV requires uint8 0/255
            m255 = (cls_mask.astype(np.uint8) * 255)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m255, connectivity=8)

            # label 0 is background
            for lab in range(1, num_labels):
                area_px = int(stats[lab, cv2.CC_STAT_AREA])
                if area_px < min_component_area_px:
                    continue

                comp = (labels == lab).astype(np.uint8)  # 0/1

                # (A) size
                s = float(area_px) / img_area

                # (C) center distance
                cx, cy = centroids[lab]  # in pixel coordinates (x,y)
                # normalize to [0,1]
                xcn = float(cx) / float(W)
                ycn = float(cy) / float(H)
                dc = math.sqrt((xcn - 0.5) ** 2 + (ycn - 0.5) ** 2)

                # (B) concavity
                conc, A, Ahull = concavity_from_component(comp)

                by_class[cls_id]["s"].append(s)
                by_class[cls_id]["c"].append(conc)
                by_class[cls_id]["dc"].append(dc)
                by_class[cls_id]["count"] += 1

                overall["s"].append(s)
                overall["c"].append(conc)
                overall["dc"].append(dc)
                overall["count"] += 1

    def fixed_hist(values, bins, vmin, vmax):
        v = np.asarray(values, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        # clip to range for safety
        v = np.clip(v, vmin, vmax)
        hist, edges = np.histogram(v, bins=bins, range=(vmin, vmax))
        return {"hist": hist.tolist(), "edges": edges.tolist()}

    # Summarize
    def summarize(name, data_dict):
        s_vals = data_dict["s"]
        c_vals = data_dict["c"]
        dc_vals = data_dict["dc"]

        summ = {
            "name": name,
            "n_components": int(data_dict["count"]),
            "size_percentiles": percentiles(s_vals),
            "concavity_percentiles": percentiles(c_vals),
            "dc_percentiles": percentiles(dc_vals),
        }

        # NEW: fixed-edge hists (comparable across classes)
        summ["size_linear_hist"] = fixed_hist(s_vals, bins=100, vmin=0.0, vmax=1.0)
        summ["concavity_hist"] = fixed_hist(c_vals, bins=50, vmin=0.0, vmax=1.0)
        summ["dc_hist"] = fixed_hist(dc_vals, bins=50, vmin=0.0, vmax=0.70710678)

        # Optional: keep old log-bins if you still want them (but not used for the main plot)
        lb = log_bins(s_vals, bins=20)
        summ["size_logbins_hist"] = {"hist": lb[0], "edges": lb[1]} if lb is not None else None

        return summ

    overall_summary = summarize("overall", overall)
    with open(os.path.join(out_dir, f"summary_overall_{split}.json"), "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)

    # per-class summary table
    rows = []
    class_summaries = {}
    for cls_id in range(len(VOC_CLASSES)):
        if (not include_background) and cls_id == 0:
            continue
        if by_class[cls_id]["count"] == 0:
            continue
        summ = summarize(VOC_CLASSES[cls_id], by_class[cls_id])
        class_summaries[VOC_CLASSES[cls_id]] = summ

        sp = summ["size_percentiles"]
        cp = summ["concavity_percentiles"]
        dp = summ["dc_percentiles"]
        rows.append({
            "class": VOC_CLASSES[cls_id],
            "n_components": summ["n_components"],
            "size_p10": sp["p10"], "size_p25": sp["p25"], "size_p50": sp["p50"], "size_p75": sp["p75"],
            "size_p90": sp["p90"],
            "conc_p10": cp["p10"], "conc_p25": cp["p25"], "conc_p50": cp["p50"], "conc_p75": cp["p75"],
            "conc_p90": cp["p90"],
            "dc_p10": dp["p10"], "dc_p25": dp["p25"], "dc_p50": dp["p50"], "dc_p75": dp["p75"], "dc_p90": dp["p90"],
        })

    df = pd.DataFrame(rows).sort_values("n_components", ascending=False)
    df.to_csv(os.path.join(out_dir, f"summary_per_class_{split}.csv"), index=False)

    with open(os.path.join(out_dir, f"summary_per_class_{split}.json"), "w", encoding="utf-8") as f:
        json.dump(class_summaries, f, indent=2)

    print(f"[DONE] Outputs written to: {os.path.abspath(out_dir)}")
    print(f" - summary_overall_{split}.json")
    print(f" - summary_per_class_{split}.csv")
    print(f" - summary_per_class_{split}.json")
    print("\nTip: Look at per_class CSV for quick comparison across classes (e.g., person vs bus).")

def _normalize_to_percent(hist):
    hist = np.asarray(hist, dtype=np.float64)
    total = hist.sum()
    if total <= 0:
        return np.zeros_like(hist)
    return hist / total * 100.0

def _normalize_prob(hist):
    hist = np.asarray(hist, dtype=np.float64)
    total = hist.sum()
    if total <= 0:
        return np.zeros_like(hist)
    return hist / total
def _bin_centers(edges):
    e = np.asarray(edges, dtype=np.float64)
    return 0.5 * (e[:-1] + e[1:])

def _plot_series(ax, series, xlabel, logy=False, xlim=None, ylim=None, as_prob=True):
    for item in series:
        x = _bin_centers(item["edges"])
        if as_prob:
            y = _normalize_prob(item["hist"])      # 0~1
        else:
            y = _normalize_prob(item["hist"]) * 100.0  # 0~100

        if logy:
            y = np.maximum(y, 1e-9)
            ax.set_yscale("log")
            ax.set_ylabel("Probability (log)" if as_prob else "Percent of masks (log)")
        else:
            ax.set_ylabel("Probability" if as_prob else "Percent of masks")

        ax.plot(x, y, linewidth=1.2, label=item["label"])

    ax.set_xlabel(xlabel)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

def plot_overall_signature(overall, plots_dir, split):
    import matplotlib.pyplot as plt

    size = overall["size_linear_hist"]
    conc = overall["concavity_hist"]
    dc   = overall["dc_hist"]

    fig = plt.figure(figsize=(14, 3.6), dpi=200)
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)

    _plot_series(ax1, [{"label":"VOC2012 (overall)", **size}],
                xlabel="Relative segmentation mask size", logy=False, xlim=(0,1))
    _plot_series(ax2, [{"label":"VOC2012 (overall)", **conc}],
                xlabel="Concavity", logy=False, xlim=(0,1))
    _plot_series(ax3, [{"label":"VOC2012 (overall)", **dc}],
                xlabel="Center distance $d_c$", logy=False, xlim=(0,0.70710678))

    # single legend top-center
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0,0,1,0.90])

    fig.savefig(os.path.join(plots_dir, f"overall_signature_{split}.png"), bbox_inches="tight")
    plt.close(fig)

def plot_all_classes_overlay(per_class, plots_dir, split, topk=20):
    import matplotlib.pyplot as plt

    # sort by sample count
    items = [(k, v.get("n_components", 0), v) for k, v in per_class.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:topk]

    size_series, conc_series, dc_series = [], [], []
    for cls, n, summ in items:
        if "size_linear_hist" not in summ:
            continue
        size_series.append({"label": cls, **summ["size_linear_hist"]})
        conc_series.append({"label": cls, **summ["concavity_hist"]})
        dc_series.append({"label": cls, **summ["dc_hist"]})

    fig = plt.figure(figsize=(14, 3.6), dpi=220)
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)

    _plot_series(ax1, size_series, "Relative segmentation mask size", logy=False, xlim=(0,1))
    _plot_series(ax2, conc_series, "Concavity", logy=False, xlim=(0,1))
    _plot_series(ax3, dc_series, "Center distance $d_c$", logy=False, xlim=(0,0.70710678))

    # Put legend outside to avoid clutter
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5),
               frameon=False, fontsize=8)

    fig.tight_layout(rect=[0,0,0.86,1])  # leave space for legend
    fig.savefig(os.path.join(plots_dir, f"all_classes_overlay_{split}.png"), bbox_inches="tight")
    plt.close(fig)

def plot_per_class_panels(per_class, plots_dir, split, topk=20):
    import matplotlib.pyplot as plt

    items = [(k, v.get("n_components", 0), v) for k, v in per_class.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:topk]

    for cls, n, summ in items:
        size = summ["size_linear_hist"]
        conc = summ["concavity_hist"]
        dc   = summ["dc_hist"]

        fig = plt.figure(figsize=(14, 3.6), dpi=200)
        ax1 = fig.add_subplot(1,3,1)
        ax2 = fig.add_subplot(1,3,2)
        ax3 = fig.add_subplot(1,3,3)

        _plot_series(ax1, [{"label": f"{cls} (n={n})", **size}],
                    "Relative segmentation mask size", logy=False, xlim=(0,1))
        _plot_series(ax2, [{"label": f"{cls} (n={n})", **conc}],
                    "Concavity", logy=False, xlim=(0,1))
        _plot_series(ax3, [{"label": f"{cls} (n={n})", **dc}],
                    "Center distance $d_c$", logy=False, xlim=(0,0.70710678))

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.tight_layout(rect=[0,0,1,0.90])

        safe = cls.replace("/", "_").replace(" ", "_")
        fig.savefig(os.path.join(plots_dir, f"class_signature_{safe}_{split}.png"), bbox_inches="tight")
        plt.close(fig)


def _safe_load_json(p):
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing summary file: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
def plot_voc_signatures_from_saved(out_dir: str, split: str = "trainval", topk_classes: int = 20):
    import os
    import json

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    overall = _safe_load_json(os.path.join(out_dir, f"summary_overall_{split}.json"))
    per_class = _safe_load_json(os.path.join(out_dir, f"summary_per_class_{split}.json"))

    # REQUIRE: use the new fixed hist fields
    if "size_linear_hist" not in overall:
        raise RuntimeError(
            "Your saved summaries do not include fixed-edge hists (size_linear_hist). "
            "Please re-run analyze_voc() after updating summarize()."
        )

    plot_overall_signature(overall, plots_dir, split)
    plot_all_classes_overlay(per_class, plots_dir, split, topk=topk_classes)
    plot_per_class_panels(per_class, plots_dir, split, topk=topk_classes)

    print(f"[DONE] Saved plots to: {os.path.abspath(plots_dir)}")


if __name__ == "__main__":
    VOC_ROOT = r"D:\uwb thesis\RelatedData\archive\VOC2012_train_val\VOC2012_train_val"
    OUT_DIR = "voc2012_signature_out"
    analyze_voc(
        voc_root=VOC_ROOT,
        split="trainval",
        min_component_area_px=20,
        include_background=False,
        out_dir="voc2012_signature_out",
    )

    plot_voc_signatures_from_saved(out_dir=OUT_DIR, split="trainval", topk_classes=20)
