import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import pycocotools.mask as mask_util

EPS = 1e-8

GT_DIR = Path("GT_masks")
JSON_DIR = Path("outputs/AllMasks_v2_score_record")
OUT_DIR = Path("outputs/carseat_param_search")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_W = np.array([0.15, 0.20, 0.40, 0.25], dtype=float)
DEFAULT_Q_BORDER = 0.30
DEFAULT_T_AREA = 0.25

GT_RATIO = 0.20
SEED = 0
N_RANDOM = 2000


def compute_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def rle_to_mask(rle_obj):
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


def find_gt_path(stem):
    for ext in [".png", ".jpg", ".jpeg", ".json"]:
        p = GT_DIR / f"{stem}_gt{ext}"
        if p.exists():
            return p
    return None


def load_gt(path):
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return rle_to_mask(data["segmentation_rle"])

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


def compute_distance_transform(h, w):
    border_mask = np.zeros((h, w), np.uint8)
    border_mask[1:-1, 1:-1] = 1
    return cv2.distanceTransform(border_mask, cv2.DIST_L2, 5)


def area_term_parabola(x, d):
    d = float(np.clip(d, 1e-6, 1.0))
    x = float(np.clip(x, 0.0, 1.0))
    return float(np.clip(1.0 - ((x - d) / d) ** 2, 0.0, 1.0))


def compute_silhouette_score_v2(mask):
    mask_u8 = mask.astype(np.uint8)

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
        total = float(areas.sum())
        if total <= 0:
            fragmentation = 0.0
        else:
            thr = 0.01 * total
            large = areas[areas >= thr]
            if large.size <= 1:
                fragmentation = 0.0
            else:
                fragmentation = 1.0 - float(large.max() / large.sum())

    fragmentation = float(np.clip(fragmentation, 0.0, 1.0))

    bg = (1 - mask_u8).astype(np.uint8)
    bg_pad = np.pad(bg, 1, mode="constant", constant_values=1)

    h, w = bg_pad.shape
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    bg_ff = bg_pad.copy()
    cv2.floodFill(bg_ff, ff_mask, (0, 0), 0)

    holes = bg_ff[1:-1, 1:-1] == 1
    hole_area = float(np.count_nonzero(holes))
    hole_ratio = float(np.clip(hole_area / area, 0.0, 1.0))

    sil = solidity * (1.0 - fragmentation) * (1.0 - hole_ratio)
    return float(np.clip(sil, 0.0, 1.0))


def mask_features(mask, W, H, dist_map, d_max, q_border, t_area):
    area_px = int(mask.sum())
    if area_px <= 0:
        return np.array([0, 0, 0, 0], dtype=float)

    img_area = W * H

    A_raw = area_px / img_area
    A = area_term_parabola(A_raw, t_area)

    ys, xs = np.where(mask)
    mx, my = xs.mean(), ys.mean()

    cx, cy = W / 2, H / 2
    Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
    C = 1.0 - float(np.clip(Dp, 0.0, 1.0))

    q = float(np.quantile(dist_map[mask], q_border))
    E = float(np.clip(q / d_max, 0.0, 1.0))

    Sil = compute_silhouette_score_v2(mask)

    return np.array([A, C, E, Sil], dtype=float)


def load_dataset():
    dataset = []

    for json_path in tqdm(sorted(JSON_DIR.glob("img_*_result.json")), desc="Loading dataset"):
        stem = json_path.name.replace("_result.json", "")
        gt_path = find_gt_path(stem)

        if gt_path is None:
            continue

        gt = load_gt(gt_path)
        H, W = gt.shape

        data = json.loads(json_path.read_text(encoding="utf-8"))
        annotations = data.get("annotations", [])

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
                    continue

            masks.append(m)
            sam_scores.append(float(ann.get("score_sam", 0.0)))
            mask_indices.append(int(ann.get("mask_index", len(mask_indices))))

        if not masks:
            continue

        dist_map = compute_distance_transform(H, W)
        d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

        dataset.append({
            "stem": stem,
            "W": W,
            "H": H,
            "gt": gt,
            "masks": masks,
            "sam_scores": np.asarray(sam_scores, dtype=float),
            "mask_indices": mask_indices,
            "dist_map": dist_map,
            "d_max": d_max,
        })

    dataset.sort(key=lambda x: x["stem"])
    return dataset


def eval_params(dataset, w_raw, q_border, t_area):
    w = normalize_weights(w_raw)
    q_border = float(np.clip(q_border, 0.0, 1.0))
    t_area = float(np.clip(t_area, 0.01, 0.95))

    ious = []

    for item in dataset:
        X = np.stack([
            mask_features(
                m,
                item["W"],
                item["H"],
                item["dist_map"],
                item["d_max"],
                q_border,
                t_area,
            )
            for m in item["masks"]
        ])

        scores = X @ w
        best = int(np.argmax(scores))
        ious.append(compute_iou(item["masks"][best], item["gt"]))

    return float(np.mean(ious)) if ious else 0.0


def eval_sam(dataset):
    ious = []
    for item in dataset:
        best = int(np.argmax(item["sam_scores"]))
        ious.append(compute_iou(item["masks"][best], item["gt"]))
    return float(np.mean(ious)) if ious else 0.0


def random_search(train_set):
    rng = np.random.default_rng(SEED)

    best = {
        "mean_iou": -1.0,
        "w": None,
        "q_border": None,
        "t_area": None,
    }

    for _ in tqdm(range(N_RANDOM), desc="Random search"):
        w = rng.dirichlet(np.ones(4))
        q_border = rng.uniform(0.05, 0.95)
        t_area = rng.uniform(0.05, 0.95)

        mean_iou = eval_params(train_set, w, q_border, t_area)

        if mean_iou > best["mean_iou"]:
            best = {
                "mean_iou": mean_iou,
                "w": w.copy(),
                "q_border": float(q_border),
                "t_area": float(t_area),
            }

    return best


def evaluate_detailed(dataset, w_raw, q_border, t_area, tag):
    w = normalize_weights(w_raw)

    records = []

    for item in dataset:
        baseline_idx = int(np.argmax(item["sam_scores"]))
        baseline_iou = compute_iou(item["masks"][baseline_idx], item["gt"])

        X = np.stack([
            mask_features(
                m,
                item["W"],
                item["H"],
                item["dist_map"],
                item["d_max"],
                q_border,
                t_area,
            )
            for m in item["masks"]
        ])

        scores = X @ w
        selected_idx = int(np.argmax(scores))
        selected_iou = compute_iou(item["masks"][selected_idx], item["gt"])

        records.append({
            "image": item["stem"],
            "setting": tag,
            "baseline_iou": baseline_iou,
            "selected_iou": selected_iou,
            "delta_vs_sam": selected_iou - baseline_iou,
            "baseline_mask_index": item["mask_indices"][baseline_idx],
            "selected_mask_index": item["mask_indices"][selected_idx],
            "selected_score": float(scores[selected_idx]),
            "wA": float(w[0]),
            "wC": float(w[1]),
            "wE": float(w[2]),
            "wSil": float(w[3]),
            "q_border": float(q_border),
            "t_area": float(t_area),
            "selected_A": float(X[selected_idx, 0]),
            "selected_C": float(X[selected_idx, 1]),
            "selected_E": float(X[selected_idx, 2]),
            "selected_Sil": float(X[selected_idx, 3]),
        })

    return pd.DataFrame(records)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    dataset = load_dataset()

    if not dataset:
        print("[ERROR] No valid data loaded.")
        return

    n_total = len(dataset)
    n_train = max(1, int(round(n_total * GT_RATIO)))

    shuffled = dataset.copy()
    random.shuffle(shuffled)

    train_set = sorted(shuffled[:n_train], key=lambda x: x["stem"])
    test_set = sorted(shuffled[n_train:], key=lambda x: x["stem"])

    print(f"\nTotal valid samples: {n_total}")
    print(f"GT used for search: {len(train_set)}")
    print(f"Held-out samples: {len(test_set)}")

    sam_all = eval_sam(dataset)
    default_train = eval_params(train_set, DEFAULT_W, DEFAULT_Q_BORDER, DEFAULT_T_AREA)
    default_all = eval_params(dataset, DEFAULT_W, DEFAULT_Q_BORDER, DEFAULT_T_AREA)

    best = random_search(train_set)

    opt_train = eval_params(train_set, best["w"], best["q_border"], best["t_area"])
    opt_all = eval_params(dataset, best["w"], best["q_border"], best["t_area"])
    opt_test = eval_params(test_set, best["w"], best["q_border"], best["t_area"]) if test_set else None
    default_test = eval_params(test_set, DEFAULT_W, DEFAULT_Q_BORDER, DEFAULT_T_AREA) if test_set else None
    sam_test = eval_sam(test_set) if test_set else None

    result = {
        "n_total": n_total,
        "n_used_for_search": len(train_set),
        "n_heldout": len(test_set),
        "seed": SEED,
        "gt_ratio": GT_RATIO,
        "n_random": N_RANDOM,

        "sam_all_mean_iou": sam_all,
        "sam_test_mean_iou": sam_test,

        "default": {
            "w": normalize_weights(DEFAULT_W).tolist(),
            "q_border": DEFAULT_Q_BORDER,
            "t_area": DEFAULT_T_AREA,
            "train_mean_iou": default_train,
            "test_mean_iou": default_test,
            "all_mean_iou": default_all,
        },

        "optimized": {
            "w": normalize_weights(best["w"]).tolist(),
            "q_border": best["q_border"],
            "t_area": best["t_area"],
            "search_train_mean_iou": opt_train,
            "test_mean_iou": opt_test,
            "all_mean_iou": opt_all,
        },

        "train_images": [x["stem"] for x in train_set],
        "test_images": [x["stem"] for x in test_set],
    }

    (OUT_DIR / "carseat_param_search_result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    df_default = evaluate_detailed(dataset, DEFAULT_W, DEFAULT_Q_BORDER, DEFAULT_T_AREA, "default")
    df_opt = evaluate_detailed(dataset, best["w"], best["q_border"], best["t_area"], "optimized")

    df = pd.concat([df_default, df_opt], ignore_index=True)
    df.to_csv(OUT_DIR / "carseat_param_search_per_image.csv", index=False)

    summary_df = pd.DataFrame([
        {
            "setting": "sam_baseline",
            "train_mean_iou": eval_sam(train_set),
            "test_mean_iou": sam_test,
            "all_mean_iou": sam_all,
        },
        {
            "setting": "default",
            "train_mean_iou": default_train,
            "test_mean_iou": default_test,
            "all_mean_iou": default_all,
        },
        {
            "setting": "optimized",
            "train_mean_iou": opt_train,
            "test_mean_iou": opt_test,
            "all_mean_iou": opt_all,
        },
    ])

    summary_df.to_csv(OUT_DIR / "carseat_param_search_summary.csv", index=False)

    print("\nBest optimized params:")
    print(f"wA   = {best['w'][0]:.6f}")
    print(f"wC   = {best['w'][1]:.6f}")
    print(f"wE   = {best['w'][2]:.6f}")
    print(f"wSil = {best['w'][3]:.6f}")
    print(f"q_border = {best['q_border']:.6f}")
    print(f"t_area   = {best['t_area']:.6f}")

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    print("\nSaved to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()