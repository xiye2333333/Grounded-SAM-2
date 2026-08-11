import os
import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2


# ============================================================
# DAVIS helpers
# ============================================================

def find_davis_root(davis_root: str) -> str:
    """
    Expect structure:
      DAVIS/
        JPEGImages/{480p,1080p}/<seq>/*.jpg
        Annotations/{480p,1080p}/<seq>/*.png (or jpg)
        ImageSets/{480p,1080p}/...
    """
    required = ["JPEGImages", "Annotations", "ImageSets"]
    for r in [davis_root, os.path.join(davis_root, "DAVIS")]:
        if all(os.path.isdir(os.path.join(r, k)) for k in required):
            return r
    raise FileNotFoundError(
        f"Could not locate DAVIS root under: {davis_root}\n"
        f"Expected folders: JPEGImages/, Annotations/, ImageSets/"
    )


def list_sequences(davis_root: str, resolution: str = "480p"):
    img_root = os.path.join(davis_root, "JPEGImages", resolution)
    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"Missing DAVIS images folder: {img_root}")
    seqs = [d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))]
    seqs.sort()
    return seqs


def list_frames_for_sequence(davis_root: str, resolution: str, seq: str):
    seq_dir = os.path.join(davis_root, "JPEGImages", resolution, seq)
    frames = [f for f in os.listdir(seq_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    frames.sort()
    return frames


def davis_mask_path(davis_root: str, resolution: str, seq: str, frame_file: str):
    """
    DAVIS GT is usually .png with same stem as RGB.
    But user mentioned .jpg; we'll try .png then .jpg.
    """
    stem = os.path.splitext(frame_file)[0]
    cand1 = os.path.join(davis_root, "Annotations", resolution, seq, f"{stem}.png")
    cand2 = os.path.join(davis_root, "Annotations", resolution, seq, f"{stem}.jpg")
    cand3 = os.path.join(davis_root, "Annotations", resolution, seq, f"{stem}.jpeg")
    for c in [cand1, cand2, cand3]:
        if os.path.isfile(c):
            return c
    return None


def load_davis_binary_mask(mask_path: str) -> np.ndarray:
    """
    Return uint8 mask in {0,1}, foreground is >0.
    Works for:
      - palette masks with indices
      - grayscale 0/255
      - RGB masks (rare): treat any nonzero pixel as FG
    """
    m = np.array(Image.open(mask_path))
    if m.ndim == 2:
        fg = (m > 0)
    else:
        fg = (m.sum(axis=2) > 0)
    return fg.astype(np.uint8)


# ============================================================
# Metrics (same as your VOC code)
# ============================================================

def concavity_from_component(comp_mask: np.ndarray):
    """
    comp_mask: uint8 {0,1} for a single connected component.
    Returns c = 1 - A/A_hull, and (A, A_hull).
    """
    A = int(comp_mask.sum())
    if A < 3:
        return 0.0, A, max(A, 1)

    m255 = (comp_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, A, max(A, 1)

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 3:
        return 0.0, A, max(A, 1)

    hull = cv2.convexHull(cnt)
    Ahull = float(cv2.contourArea(hull))
    if Ahull <= 1e-6:
        Ahull = float(A)

    ratio = float(A) / float(Ahull)
    ratio = min(max(ratio, 0.0), 1.0)
    c = 1.0 - ratio
    return float(c), A, float(Ahull)


def log_bins(values, bins=20):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    v = np.clip(v, 1e-8, 1.0)
    edges = np.logspace(np.log10(1e-8), 0, bins + 1)
    hist, edges = np.histogram(v, bins=edges)
    return hist.tolist(), edges.tolist()


def percentiles(values, ps=(10, 25, 50, 75, 90)):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"p{p}": None for p in ps}
    out = np.percentile(v, ps)
    return {f"p{p}": float(x) for p, x in zip(ps, out)}


def fixed_hist(values, bins, vmin, vmax):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    v = np.clip(v, vmin, vmax)
    hist, edges = np.histogram(v, bins=bins, range=(vmin, vmax))
    return {"hist": hist.tolist(), "edges": edges.tolist()}


# ============================================================
# Analysis for DAVIS (sequence as "class")
# ============================================================

def analyze_davis(
    davis_root: str,
    resolution: str = "480p",
    min_component_area_px: int = 20,
    include_background: bool = False,   # kept for compatibility; DAVIS is FG/BG so usually False
    out_dir: str = None,
    split_file: str = None,             # optional: DAVIS/ImageSets/... txt
):
    davis_root = find_davis_root(davis_root)

    if out_dir is None:
        out_dir = os.path.join(davis_root, "Analyze")
    os.makedirs(out_dir, exist_ok=True)

    # sequences
    if split_file is not None:
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file, "r", encoding="utf-8") as f:
            seqs = [x.strip() for x in f.readlines() if x.strip()]
        seqs = sorted(seqs)
        print(f"[INFO] Using split file with {len(seqs)} sequences: {split_file}")
    else:
        seqs = list_sequences(davis_root, resolution=resolution)
        print(f"[INFO] Found {len(seqs)} sequences under JPEGImages/{resolution}/")

    # Collect per-seq samples
    by_class = defaultdict(lambda: {"s": [], "c": [], "dc": [], "count": 0})
    overall = {"s": [], "c": [], "dc": [], "count": 0}

    # Iterate
    pbar = tqdm(seqs, desc=f"Processing DAVIS {resolution}")
    for seq in pbar:
        frames = list_frames_for_sequence(davis_root, resolution, seq)
        if len(frames) == 0:
            continue

        for frame_file in frames:
            mask_path = davis_mask_path(davis_root, resolution, seq, frame_file)
            if mask_path is None:
                continue

            fg = load_davis_binary_mask(mask_path)  # {0,1}
            H, W = fg.shape[:2]
            img_area = float(H * W)

            if not fg.any():
                continue

            m255 = (fg.astype(np.uint8) * 255)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m255, connectivity=8)

            for lab in range(1, num_labels):
                area_px = int(stats[lab, cv2.CC_STAT_AREA])
                if area_px < min_component_area_px:
                    continue

                comp = (labels == lab).astype(np.uint8)  # 0/1

                # (A) size
                s = float(area_px) / img_area

                # (C) center distance
                cx, cy = centroids[lab]  # (x,y)
                xcn = float(cx) / float(W)
                ycn = float(cy) / float(H)
                dc = math.sqrt((xcn - 0.5) ** 2 + (ycn - 0.5) ** 2)

                # (B) concavity
                conc, A, Ahull = concavity_from_component(comp)

                by_class[seq]["s"].append(s)
                by_class[seq]["c"].append(conc)
                by_class[seq]["dc"].append(dc)
                by_class[seq]["count"] += 1

                overall["s"].append(s)
                overall["c"].append(conc)
                overall["dc"].append(dc)
                overall["count"] += 1

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

        summ["size_linear_hist"] = fixed_hist(s_vals, bins=100, vmin=0.0, vmax=1.0)
        summ["concavity_hist"] = fixed_hist(c_vals, bins=50, vmin=0.0, vmax=1.0)
        summ["dc_hist"] = fixed_hist(dc_vals, bins=50, vmin=0.0, vmax=0.70710678)

        lb = log_bins(s_vals, bins=20)
        summ["size_logbins_hist"] = {"hist": lb[0], "edges": lb[1]} if lb is not None else None
        return summ

    tag = resolution  # e.g., "480p"

    overall_summary = summarize("overall", overall)
    with open(os.path.join(out_dir, f"summary_overall_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)

    # per-class (sequence) summaries
    rows = []
    class_summaries = {}
    for seq, d in by_class.items():
        if d["count"] == 0:
            continue
        summ = summarize(seq, d)
        class_summaries[seq] = summ

        sp = summ["size_percentiles"]
        cp = summ["concavity_percentiles"]
        dp = summ["dc_percentiles"]
        rows.append({
            "class": seq,
            "n_components": summ["n_components"],
            "size_p10": sp["p10"], "size_p25": sp["p25"], "size_p50": sp["p50"], "size_p75": sp["p75"], "size_p90": sp["p90"],
            "conc_p10": cp["p10"], "conc_p25": cp["p25"], "conc_p50": cp["p50"], "conc_p75": cp["p75"], "conc_p90": cp["p90"],
            "dc_p10": dp["p10"], "dc_p25": dp["p25"], "dc_p50": dp["p50"], "dc_p75": dp["p75"], "dc_p90": dp["p90"],
        })

    df = pd.DataFrame(rows).sort_values("n_components", ascending=False)
    df.to_csv(os.path.join(out_dir, f"summary_per_class_{tag}.csv"), index=False)

    with open(os.path.join(out_dir, f"summary_per_class_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(class_summaries, f, indent=2)

    print(f"[DONE] Outputs written to: {os.path.abspath(out_dir)}")
    print(f" - summary_overall_{tag}.json")
    print(f" - summary_per_class_{tag}.csv")
    print(f" - summary_per_class_{tag}.json")


# ============================================================
# Plotting (reuse your plotting code, just change labels/tags)
# ============================================================

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
        y = _normalize_prob(item["hist"]) if as_prob else _normalize_prob(item["hist"]) * 100.0

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

def _safe_load_json(p):
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing summary file: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_overall_signature(overall, plots_dir, tag, dataset_label="DAVIS"):
    import matplotlib.pyplot as plt

    size = overall["size_linear_hist"]
    conc = overall["concavity_hist"]
    dc   = overall["dc_hist"]

    fig = plt.figure(figsize=(14, 3.6), dpi=200)
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)

    _plot_series(ax1, [{"label":f"{dataset_label} (overall)", **size}],
                xlabel="Relative segmentation mask size", logy=False, xlim=(0,1))
    _plot_series(ax2, [{"label":f"{dataset_label} (overall)", **conc}],
                xlabel="Concavity", logy=False, xlim=(0,1))
    _plot_series(ax3, [{"label":f"{dataset_label} (overall)", **dc}],
                xlabel="Center distance $d_c$", logy=False, xlim=(0,0.70710678))

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0,0,1,0.90])

    fig.savefig(os.path.join(plots_dir, f"overall_signature_{tag}.png"), bbox_inches="tight")
    plt.close(fig)

def plot_all_classes_overlay(per_class, plots_dir, tag, topk=20):
    import matplotlib.pyplot as plt

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

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5),
               frameon=False, fontsize=8)

    fig.tight_layout(rect=[0,0,0.86,1])
    fig.savefig(os.path.join(plots_dir, f"all_classes_overlay_{tag}.png"), bbox_inches="tight")
    plt.close(fig)

def plot_per_class_panels(per_class, plots_dir, tag, topk=20):
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
        fig.savefig(os.path.join(plots_dir, f"class_signature_{safe}_{tag}.png"), bbox_inches="tight")
        plt.close(fig)

def plot_davis_signatures_from_saved(out_dir: str, tag: str = "480p", topk_classes: int = 20):
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    overall = _safe_load_json(os.path.join(out_dir, f"summary_overall_{tag}.json"))
    per_class = _safe_load_json(os.path.join(out_dir, f"summary_per_class_{tag}.json"))

    if "size_linear_hist" not in overall:
        raise RuntimeError("Saved summaries do not include fixed-edge hists; re-run analyze_davis().")

    plot_overall_signature(overall, plots_dir, tag, dataset_label="DAVIS")
    plot_all_classes_overlay(per_class, plots_dir, tag, topk=topk_classes)
    plot_per_class_panels(per_class, plots_dir, tag, topk=topk_classes)

    print(f"[DONE] Saved plots to: {os.path.abspath(plots_dir)}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    DAVIS_ROOT = r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS"
    OUT_DIR = r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\Analyze"

    analyze_davis(
        davis_root=DAVIS_ROOT,
        resolution="480p",
        min_component_area_px=20,
        out_dir=OUT_DIR,
        split_file=None,   # 需要用 ImageSets 的 split 再填这里
    )

    plot_davis_signatures_from_saved(out_dir=OUT_DIR, tag="480p", topk_classes=20)