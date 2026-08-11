# optimize_weights_qt_with_gt_precompute.py 计算carseat数据集的最优weights，当前只用了argmax随机搜
import os, json, argparse, random
from pathlib import Path
import numpy as np
import cv2
import pycocotools.mask as mask_util
from scipy.optimize import minimize

# ----------------------------
# Optional progress bar (tqdm)
# ----------------------------
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

EPS = 1e-8

# ----------------------------
# RLE / mask utils
# ----------------------------
def decode_rle_to_bool(rle_obj):
    m = mask_util.decode(rle_obj)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)

def decode_ann_mask_bool(ann):
    return decode_rle_to_bool(ann["segmentation"])

def load_gt_mask_from_gt_json(gt_json_path: Path):
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    if "segmentation_rle" in gt and isinstance(gt["segmentation_rle"], dict):
        return decode_rle_to_bool(gt["segmentation_rle"])

    png_path = gt.get("gt_mask_png", None)
    if png_path and os.path.exists(png_path):
        m = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        return (m > 0)

    return None

def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0

def compute_distance_transform(h, w):
    border_mask = np.zeros((h, w), np.uint8)
    border_mask[1:-1, 1:-1] = 1
    return cv2.distanceTransform(border_mask, distanceType=cv2.DIST_L2, maskSize=5)

# ----------------------------
# Score terms
# ----------------------------
def area_term_parabola_vec(area_ratio: np.ndarray, d: float, eps: float = 1e-6) -> np.ndarray:
    d = float(np.clip(d, eps, 1.0))
    x = np.clip(area_ratio, 0.0, 1.0)
    val = 1.0 - ((x - d) / d) ** 2
    return np.clip(val, 0.0, 1.0)

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

def normalize_weights(w):
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= EPS:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    return w / s

# ----------------------------
# GT index
# ----------------------------
def build_gt_index(gt_dir: str):
    gt_dirp = Path(gt_dir)
    gt_files = sorted(gt_dirp.glob("*.json"))
    idx = {}
    for fp in gt_files:
        name = fp.stem  # img_0003_gt
        key = name[:-3] if name.endswith("_gt") else name
        idx[key] = fp
    return idx

def infer_img_key_from_result_json(data: dict, fp: Path):
    imgp = data.get("image_path", None)
    if imgp:
        return Path(imgp).stem
    stem = fp.stem
    if stem.endswith("_result"):
        return stem[:-7]
    return stem

# ----------------------------
# Precompute dataset
# ----------------------------
def precompute_dataset(json_dir: str, gt_dir: str, max_files=None, store_dist_dtype="float16"):
    json_files = sorted(Path(json_dir).glob("*_result.json"))
    if max_files:
        json_files = json_files[:max_files]

    gt_index = build_gt_index(gt_dir)

    dataset = []
    skipped_no_gt = 0
    skipped_bad = 0

    it = json_files
    if tqdm is not None:
        it = tqdm(it, desc="Precomputing", unit="file")

    for fp in it:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            skipped_bad += 1
            continue

        anns = data.get("annotations", [])
        if not anns:
            skipped_bad += 1
            continue

        key = infer_img_key_from_result_json(data, fp)
        gt_json_path = gt_index.get(key, None)
        if gt_json_path is None or (not gt_json_path.exists()):
            skipped_no_gt += 1
            continue

        gt_mask = load_gt_mask_from_gt_json(gt_json_path)
        if gt_mask is None:
            skipped_no_gt += 1
            continue

        H = int(data.get("img_height", gt_mask.shape[0]))
        W = int(data.get("img_width",  gt_mask.shape[1]))

        if gt_mask.shape != (H, W):
            skipped_bad += 1
            continue

        masks = [decode_ann_mask_bool(a) for a in anns]

        # shared per-image
        dist_map = compute_distance_transform(H, W)
        d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

        img_area = float(W * H)
        cx, cy = W / 2.0, H / 2.0
        norm = (0.5 * np.hypot(W, H)) + 1e-12

        K = len(masks)
        area_ratio = np.zeros((K,), dtype=np.float32)
        center = np.zeros((K,), dtype=np.float32)
        sil = np.zeros((K,), dtype=np.float32)
        iou = np.zeros((K,), dtype=np.float32)
        dist_sorted = [None] * K  # list of 1D sorted dist values inside mask

        for j, m in enumerate(masks):
            a = float(m.sum())
            if a <= 0:
                area_ratio[j] = 0.0
                center[j] = 0.0
                sil[j] = 0.0
                iou[j] = 0.0
                dist_sorted[j] = np.zeros((1,), dtype=np.float16 if store_dist_dtype=="float16" else np.float32)
                continue

            area_ratio[j] = a / img_area

            ys, xs = np.where(m)
            mx, my = float(xs.mean()), float(ys.mean())
            dp = np.hypot(mx - cx, my - cy) / norm
            center[j] = 1.0 - float(np.clip(dp, 0.0, 1.0))

            sil[j] = compute_silhouette_score_v2(m)
            iou[j] = compute_iou(m, gt_mask)

            vals = dist_map[m].astype(np.float32, copy=False)
            vals.sort()  # in-place sort
            if store_dist_dtype == "float16":
                vals = vals.astype(np.float16, copy=False)
            dist_sorted[j] = vals

        dataset.append({
            "file": fp.name,
            "key": key,
            "K": K,
            "area_ratio": area_ratio,
            "center": center,
            "sil": sil,
            "iou": iou,
            "dist_sorted": dist_sorted,
            "d_max": float(d_max),
        })

    return dataset, skipped_no_gt, skipped_bad

# ----------------------------
# Fast evaluation using precomputed arrays
# ----------------------------
def compute_E_from_sorted(dist_sorted_list, q_border: float, d_max: float) -> np.ndarray:
    # nearest-rank quantile on sorted array
    q = float(np.clip(q_border, 0.0, 1.0))
    E = np.zeros((len(dist_sorted_list),), dtype=np.float32)
    inv_dmax = 1.0 / (d_max + 1e-12)
    for i, arr in enumerate(dist_sorted_list):
        n = int(arr.shape[0])
        if n <= 0:
            E[i] = 0.0
            continue
        idx = int(round(q * (n - 1)))
        v = float(arr[idx])  # already distance value
        E[i] = float(np.clip(v * inv_dmax, 0.0, 1.0))
    return E

def eval_params_fast(dataset, w_raw, q_border, t_area):
    w = normalize_weights(w_raw)
    q_border = float(np.clip(q_border, 0.0, 1.0))
    t_area = float(np.clip(t_area, 0.0, 1.0))

    wA, wC, wB, wS = float(w[0]), float(w[1]), float(w[2]), float(w[3])

    ious = []
    for item in dataset:
        A = area_term_parabola_vec(item["area_ratio"], t_area)  # (K,)
        C = item["center"]
        S = item["sil"]
        E = compute_E_from_sorted(item["dist_sorted"], q_border, item["d_max"])

        scores = wA * A + wC * C + wB * E + wS * S
        top = int(np.argmax(scores))
        ious.append(float(item["iou"][top]))

    return float(np.mean(ious)) if ious else 0.0

# ----------------------------
# Random search + progress bar + Powell refinement
# ----------------------------
def optimize_random_fast(train, val, seed=0, n_random=15000,
                         q_range=(0.05, 0.95), t_range=(0.05, 0.95)):
    rng = np.random.default_rng(seed)

    best = None
    best_val = -1.0

    iterator = range(n_random)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Random search", unit="trial")

    for k in iterator:
        w = rng.dirichlet(alpha=np.ones(4))
        q = float(rng.uniform(q_range[0], q_range[1]))
        t = float(rng.uniform(t_range[0], t_range[1]))
        v = eval_params_fast(val, w, q, t)
        if v > best_val:
            best_val = v
            best = (w.copy(), q, t)

        if tqdm is not None and (k % 50 == 0):
            iterator.set_postfix(best_val=f"{best_val:.4f}")

    w0, q0, t0 = best

    # Local refinement
    def softplus(x):
        return np.log1p(np.exp(x))

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def pack(u):
        w_pos = softplus(u[:4])
        w = normalize_weights(w_pos)
        q = q_range[0] + (q_range[1] - q_range[0]) * sigmoid(u[4])
        t = t_range[0] + (t_range[1] - t_range[0]) * sigmoid(u[5])
        return w, float(q), float(t)

    def objective(u):
        w, q, t = pack(u)
        return -eval_params_fast(val, w, q, t)

    def inv_softplus(y):
        y = np.clip(y, 1e-6, None)
        return np.log(np.exp(y) - 1.0 + 1e-12)

    def inv_sigmoid(y):
        y = np.clip(y, 1e-6, 1 - 1e-6)
        return np.log(y / (1 - y))

    u_w = inv_softplus(w0 + 1e-6)
    qn = (q0 - q_range[0]) / (q_range[1] - q_range[0] + 1e-12)
    tn = (t0 - t_range[0]) / (t_range[1] - t_range[0] + 1e-12)
    u0 = np.concatenate([u_w, [inv_sigmoid(qn), inv_sigmoid(tn)]]).astype(float)

    # Powell is deterministic given u0; can still be slow but far cheaper now
    res = minimize(objective, u0, method="Powell", options={"maxiter": 200, "disp": False})
    w_opt, q_opt, t_opt = pack(res.x)

    train_iou = eval_params_fast(train, w_opt, q_opt, t_opt)
    val_iou = eval_params_fast(val, w_opt, q_opt, t_opt)

    return w_opt, q_opt, t_opt, train_iou, val_iou

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json_dir",
        type=str,
        default=r"D:\uwb thesis\code\Grounded-SAM-2\outputs\AllMasks_v2_score_record",
        help="Folder containing *_result.json"
    )
    ap.add_argument(
        "--gt_dir",
        type=str,
        default=r"D:\uwb thesis\code\Grounded-SAM-2\GT_masks",
        help="Folder containing img_????_gt.json and corresponding gt png/rle"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", type=float, default=0.8, help="train ratio by images")
    ap.add_argument("--n_random", type=int, default=15000)
    ap.add_argument("--q_low", type=float, default=0.05)
    ap.add_argument("--q_high", type=float, default=0.95)
    ap.add_argument("--t_low", type=float, default=0.05)
    ap.add_argument("--t_high", type=float, default=0.95)
    ap.add_argument("--max_files", type=int, default=0, help="0 means no limit")
    ap.add_argument("--dist_dtype", type=str, default="float16", choices=["float16", "float32"],
                    help="store per-mask sorted border distances as float16 to save memory")
    args = ap.parse_args()

    max_files = None if args.max_files <= 0 else args.max_files

    dataset, skipped_no_gt, skipped_bad = precompute_dataset(
        args.json_dir, args.gt_dir, max_files=max_files, store_dist_dtype=args.dist_dtype
    )
    if len(dataset) < 8:
        raise SystemExit(
            f"Too few usable samples with GT: {len(dataset)} "
            f"(skipped_no_gt={skipped_no_gt}, skipped_bad={skipped_bad})."
        )

    rnd = random.Random(args.seed)
    rnd.shuffle(dataset)
    n_train = int(len(dataset) * args.split)
    train = dataset[:n_train]
    val = dataset[n_train:]

    q_range = (min(args.q_low, args.q_high), max(args.q_low, args.q_high))
    t_range = (min(args.t_low, args.t_high), max(args.t_low, args.t_high))

    # Baseline
    w_eq = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    q0 = float(np.clip(0.40, *q_range))
    t0 = float(np.clip(0.30, *t_range))

    print(f"Loaded usable images: {len(dataset)} | train={len(train)} val={len(val)}")
    print(f"Skipped: no_gt={skipped_no_gt}, bad_json_or_shape={skipped_bad}")
    print(f"Search ranges: q_border in {q_range}, t_area in {t_range}")
    print(f"[Baseline] w={w_eq}, q={q0:.3f}, t={t0:.3f}")
    print(f"  train meanIoU={eval_params_fast(train, w_eq, q0, t0):.4f}")
    print(f"  val   meanIoU={eval_params_fast(val,   w_eq, q0, t0):.4f}")

    w_opt, q_opt, t_opt, train_iou, val_iou = optimize_random_fast(
        train, val, seed=args.seed, n_random=args.n_random, q_range=q_range, t_range=t_range
    )

    print("\n=== Direct Random Search + Powell Refinement (FAST) ===")
    print(f"w_opt (A,C,E,Sil) = {w_opt} (sum={w_opt.sum():.3f})")
    print(f"q_border          = {q_opt:.4f}")
    print(f"t_area            = {t_opt:.4f}")
    print(f"train meanIoU     = {train_iou:.4f}")
    print(f"val   meanIoU     = {val_iou:.4f}")

if __name__ == "__main__":
    main()
