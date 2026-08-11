import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import pycocotools.mask as mask_util

EPS = 1e-8


# =========================================================
# Paths
# =========================================================

IMAGE_DIR = Path("assets/images")
GT_DIR = Path("GT_masks")
JSON_DIR = Path("outputs/AllMasks_v2_score_record")
OUT_DIR = Path("outputs/carseat_ablation")

OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Basic utils
# =========================================================

def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def rle_to_mask(rle_obj: dict) -> np.ndarray:
    rle = {
        "size": rle_obj["size"],
        "counts": rle_obj["counts"].encode("utf-8")
        if isinstance(rle_obj["counts"], str)
        else rle_obj["counts"],
    }
    m = mask_util.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)


def find_gt_path(stem: str, gt_dir: Path):
    candidates = [
        gt_dir / f"{stem}_gt.png",
        gt_dir / f"{stem}_gt.jpg",
        gt_dir / f"{stem}_gt.jpeg",
        gt_dir / f"{stem}_gt.json",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def load_gt_from_json(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "segmentation_rle" not in data:
        raise KeyError(f"Missing segmentation_rle in GT json: {path}")

    return rle_to_mask(data["segmentation_rle"])


def load_gt_bool_auto(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".json":
        return load_gt_from_json(path)

    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if m is None:
        raise FileNotFoundError(path)

    return (m > 127)


def normalize_weights(w):
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())

    if s <= EPS:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)

    return w / s


# =========================================================
# Feature functions
# =========================================================

def compute_distance_transform(h: int, w: int):
    border_mask = np.zeros((h, w), np.uint8)
    border_mask[1:-1, 1:-1] = 1

    return cv2.distanceTransform(
        border_mask,
        distanceType=cv2.DIST_L2,
        maskSize=5,
    )


def area_term_parabola(x: float, d: float, eps: float = 1e-6) -> float:
    d = float(np.clip(d, eps, 1.0))
    x = float(np.clip(x, 0.0, 1.0))

    val = 1.0 - ((x - d) / d) ** 2

    return float(np.clip(val, 0.0, 1.0))


def compute_silhouette_score_v2(mask: np.ndarray) -> float:
    mask_u8 = (mask.astype(np.uint8) > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return 0.0

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))

    if area <= 0:
        return 0.0

    hull = cv2.convexHull(largest)
    hull_area = float(cv2.contourArea(hull))

    solidity = 0.0 if hull_area <= 0 else area / hull_area
    solidity = float(np.clip(solidity, 0.0, 1.0))

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )

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
                fragmentation = 1.0 - largest_area / float(np.sum(large_areas))

    fragmentation = float(np.clip(fragmentation, 0.0, 1.0))

    bg = (1 - mask_u8).astype(np.uint8)
    bg_pad = np.pad(bg, pad_width=1, mode="constant", constant_values=1)

    h, w = bg_pad.shape
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    bg_ff = bg_pad.copy()
    cv2.floodFill(bg_ff, ff_mask, (0, 0), 0)

    holes_map = bg_ff[1:-1, 1:-1] == 1
    holes_area = float(np.count_nonzero(holes_map))

    hole_ratio = holes_area / area if area > 0 else 0.0
    hole_ratio = float(np.clip(hole_ratio, 0.0, 1.0))

    sil = solidity * (1.0 - fragmentation) * (1.0 - hole_ratio)

    return float(np.clip(sil, 0.0, 1.0))


def mask_features(
    mask_bool: np.ndarray,
    W: int,
    H: int,
    dist_map: np.ndarray,
    d_max: float,
    q_border: float,
    t_area: float,
):
    img_area = W * H
    area_px = int(mask_bool.sum())

    if area_px <= 0:
        return np.array([0, 0, 0, 0], dtype=float)

    # A: area plausibility
    A_raw = area_px / img_area
    A = area_term_parabola(A_raw, t_area)

    # C: center prior
    ys, xs = np.where(mask_bool)

    mx = xs.mean()
    my = ys.mean()

    cx = W / 2
    cy = H / 2

    Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
    C = 1.0 - float(np.clip(Dp, 0.0, 1.0))

    # E: boundary / interior support
    q = float(np.quantile(dist_map[mask_bool], q_border))
    E = float(np.clip(q / d_max, 0.0, 1.0))

    # Sil: silhouette coherence
    Sil = compute_silhouette_score_v2(mask_bool)

    return np.array([A, C, E, Sil], dtype=float)


# =========================================================
# Ablation config
# =========================================================

Q_BORDER = 0.4
T_AREA = 0.35

ABLATIONS = {
    # full / remove-one
    "default_full": np.array([0.15, 0.20, 0.40, 0.25], dtype=float),
    "no_area": np.array([0.00, 0.20, 0.40, 0.25], dtype=float),
    "no_center": np.array([0.15, 0.00, 0.40, 0.25], dtype=float),
    "no_border": np.array([0.15, 0.20, 0.00, 0.25], dtype=float),
    "no_silhouette": np.array([0.15, 0.20, 0.40, 0.00], dtype=float),

    # single-term
    "only_area": np.array([1.00, 0.00, 0.00, 0.00], dtype=float),
    "only_center": np.array([0.00, 1.00, 0.00, 0.00], dtype=float),
    "only_border": np.array([0.00, 0.00, 1.00, 0.00], dtype=float),
    "only_silhouette": np.array([0.00, 0.00, 0.00, 1.00], dtype=float),
}
#     "W_AREA": 0.2,
#     "W_CENTER": 0.2,
#     "W_BORDER": 0.6,
#     "Q_BORDER": 0.25,

# =========================================================
# Main
# =========================================================

def main():
    json_files = sorted(JSON_DIR.glob("img_*_result.json"))

    if not json_files:
        print(f"[ERROR] No result json files found in: {JSON_DIR}")
        return

    all_records = []
    skipped_missing_gt = 0
    skipped_no_annotations = 0
    skipped_shape_mismatch = 0
    skipped_no_valid_masks = 0

    for json_path in tqdm(json_files, desc="CarSeat ablation"):
        stem = json_path.name.replace("_result.json", "")

        gt_path = find_gt_path(stem, GT_DIR)

        if gt_path is None:
            skipped_missing_gt += 1
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        annotations = data.get("annotations", [])

        if not annotations:
            skipped_no_annotations += 1
            continue

        gt = load_gt_bool_auto(gt_path)
        H, W = gt.shape

        masks = []
        sam_scores = []
        mask_indices = []

        for ann in annotations:
            if "segmentation" not in ann:
                continue

            m = rle_to_mask(ann["segmentation"])

            if m.shape != gt.shape:
                if m.T.shape == gt.shape:
                    m = m.T
                else:
                    skipped_shape_mismatch += 1
                    print(
                        f"[WARN] Shape mismatch in {stem}: "
                        f"mask={m.shape}, gt={gt.shape}, gt_path={gt_path}"
                    )
                    continue

            masks.append(m)
            sam_scores.append(float(ann.get("score_sam", 0.0)))
            mask_indices.append(int(ann.get("mask_index", len(mask_indices))))

        if len(masks) == 0:
            skipped_no_valid_masks += 1
            continue

        sam_scores = np.asarray(sam_scores, dtype=float)

        # SAM baseline: candidate with highest SAM confidence
        baseline_idx_local = int(np.argmax(sam_scores))
        baseline_mask = masks[baseline_idx_local]
        baseline_iou = compute_iou(baseline_mask, gt)
        baseline_mask_index = mask_indices[baseline_idx_local]

        dist_map = compute_distance_transform(H, W)
        d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

        X = np.stack(
            [
                mask_features(
                    m,
                    W,
                    H,
                    dist_map,
                    d_max,
                    Q_BORDER,
                    T_AREA,
                )
                for m in masks
            ],
            axis=0,
        )

        for ablation_name, w_raw in ABLATIONS.items():
            w = normalize_weights(w_raw)

            feature_scores = X @ w
            selected_idx_local = int(np.argmax(feature_scores))

            selected_mask = masks[selected_idx_local]
            selected_iou = compute_iou(selected_mask, gt)
            selected_mask_index = mask_indices[selected_idx_local]

            all_records.append(
                {
                    "image": stem,
                    "gt_path": str(gt_path),
                    "ablation": ablation_name,
                    "n_candidates": len(masks),

                    "baseline_mask_index": baseline_mask_index,
                    "baseline_sam_score": float(sam_scores[baseline_idx_local]),
                    "baseline_iou": float(baseline_iou),

                    "selected_mask_index": selected_mask_index,
                    "selected_feature_score": float(feature_scores[selected_idx_local]),
                    "selected_iou": float(selected_iou),
                    "delta_vs_sam": float(selected_iou - baseline_iou),

                    "wA": float(w[0]),
                    "wC": float(w[1]),
                    "wE": float(w[2]),
                    "wSil": float(w[3]),
                    "q_border": float(Q_BORDER),
                    "t_area": float(T_AREA),

                    "selected_A": float(X[selected_idx_local, 0]),
                    "selected_C": float(X[selected_idx_local, 1]),
                    "selected_E": float(X[selected_idx_local, 2]),
                    "selected_Sil": float(X[selected_idx_local, 3]),
                }
            )

    df = pd.DataFrame(all_records)

    if df.empty:
        print("[ERROR] No valid records were produced.")
        print(f"json_files: {len(json_files)}")
        print(f"skipped_missing_gt: {skipped_missing_gt}")
        print(f"skipped_no_annotations: {skipped_no_annotations}")
        print(f"skipped_shape_mismatch: {skipped_shape_mismatch}")
        print(f"skipped_no_valid_masks: {skipped_no_valid_masks}")
        return

    per_image_csv = OUT_DIR / "carseat_ablation_per_image.csv"
    df.to_csv(per_image_csv, index=False)

    summary_records = []

    for ablation_name, g in df.groupby("ablation"):
        baseline_mean = float(g["baseline_iou"].mean())
        selected_mean = float(g["selected_iou"].mean())

        summary_records.append(
            {
                "ablation": ablation_name,
                "n_images": int(g["image"].nunique()),
                "n_records": int(len(g)),
                "baseline_mean_iou": baseline_mean,
                "selected_mean_iou": selected_mean,
                "mean_delta_vs_sam": selected_mean - baseline_mean,
                "mean_delta_per_image": float(g["delta_vs_sam"].mean()),
                "improved_count": int((g["delta_vs_sam"] > 1e-6).sum()),
                "unchanged_count": int((np.abs(g["delta_vs_sam"]) <= 1e-6).sum()),
                "worsened_count": int((g["delta_vs_sam"] < -1e-6).sum()),
            }
        )

    summary_df = pd.DataFrame(summary_records)

    ablation_order = [
        "default_full",
        "no_area",
        "no_center",
        "no_border",
        "no_silhouette",
        "only_area",
        "only_center",
        "only_border",
        "only_silhouette",
    ]

    summary_df["ablation"] = pd.Categorical(
        summary_df["ablation"],
        categories=ablation_order,
        ordered=True,
    )
    summary_df = summary_df.sort_values("ablation")

    summary_csv = OUT_DIR / "carseat_ablation_summary.csv"
    summary_json = OUT_DIR / "carseat_ablation_summary.json"

    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(summary_records, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(per_image_csv)
    print(summary_csv)
    print(summary_json)

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    print("\nSkipped:")
    print(f"missing_gt: {skipped_missing_gt}")
    print(f"no_annotations: {skipped_no_annotations}")
    print(f"shape_mismatch: {skipped_shape_mismatch}")
    print(f"no_valid_masks: {skipped_no_valid_masks}")


if __name__ == "__main__":
    main()