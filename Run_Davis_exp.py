
import os, json, random, argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

import torch
from torchvision.ops import box_convert
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import pycocotools.mask as mask_util
from scipy.optimize import minimize

EPS = 1e-8


# =========================================================
# COCO RLE utils
# =========================================================
def mask_to_rle(mask_bool: np.ndarray) -> Dict[str, Any]:
    """
    mask_bool: HxW bool/uint8
    returns COCO RLE dict with counts as str.
    """
    m = np.asfortranarray(mask_bool.astype(np.uint8))
    rle = mask_util.encode(m)
    rle["counts"] = rle["counts"].decode("utf-8")
    return {
        "size": [int(rle["size"][0]), int(rle["size"][1])],
        "counts": rle["counts"],
    }


def rle_to_mask(rle_obj: Dict[str, Any]) -> np.ndarray:
    rle = {
        "size": rle_obj["size"],
        "counts": rle_obj["counts"].encode("utf-8"),
    }
    m = mask_util.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)


def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


# =========================================================
# DAVIS paths
# =========================================================
def find_davis_root(davis_root: str) -> str:
    required = ["JPEGImages", "Annotations", "ImageSets"]
    for r in [davis_root, os.path.join(davis_root, "DAVIS")]:
        if all(os.path.isdir(os.path.join(r, k)) for k in required):
            return r
    raise FileNotFoundError(
        f"Could not locate DAVIS root under: {davis_root}\n"
        f"Expected: JPEGImages/, Annotations/, ImageSets/"
    )


def list_sequences(davis_root: str, resolution: str) -> List[str]:
    p = os.path.join(davis_root, "JPEGImages", resolution)
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Missing: {p}")
    seqs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
    seqs.sort()
    return seqs


def list_frames(davis_root: str, resolution: str, seq: str) -> List[str]:
    p = os.path.join(davis_root, "JPEGImages", resolution, seq)
    frames = [f for f in os.listdir(p) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    frames.sort()
    return frames


def davis_img_path(davis_root: str, resolution: str, seq: str, frame_file: str) -> str:
    return os.path.join(davis_root, "JPEGImages", resolution, seq, frame_file)


def davis_gt_path(davis_root: str, resolution: str, seq: str, frame_file: str) -> Optional[str]:
    stem = os.path.splitext(frame_file)[0]
    candidates = [
        os.path.join(davis_root, "Annotations", resolution, seq, f"{stem}.png"),
        os.path.join(davis_root, "Annotations", resolution, seq, f"{stem}.jpg"),
        os.path.join(davis_root, "Annotations", resolution, seq, f"{stem}.jpeg"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def load_davis_gt_bool(gt_path: str) -> np.ndarray:
    m = np.array(Image.open(gt_path))
    if m.ndim == 2:
        fg = (m > 0)
    else:
        fg = (m.sum(axis=2) > 0)
    return fg.astype(bool)


# =========================================================
# Feature functions
#   [A, C, E, Sil]
# =========================================================
def compute_distance_transform(h, w):
    border_mask = np.zeros((h, w), np.uint8)
    border_mask[1:-1, 1:-1] = 1
    return cv2.distanceTransform(border_mask, distanceType=cv2.DIST_L2, maskSize=5)


def area_term_parabola(x: float, d: float, eps: float = 1e-6) -> float:
    d = float(np.clip(d, eps, 1.0))
    x = float(np.clip(x, 0.0, 1.0))
    val = 1.0 - ((x - d) / d) ** 2
    return float(np.clip(val, 0.0, 1.0))


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


def mask_features(mask_bool, W, H, dist_map, d_max, q_border, t_area):
    img_area = W * H
    cx, cy = W / 2, H / 2

    area_px = int(mask_bool.sum())
    if area_px <= 0:
        return np.array([0, 0, 0, 0], dtype=float)

    # A
    A_raw = area_px / img_area
    A = area_term_parabola(A_raw, t_area)

    # C
    ys, xs = np.where(mask_bool)
    mx, my = xs.mean(), ys.mean()
    Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
    C = 1.0 - float(np.clip(Dp, 0.0, 1.0))

    # E
    q = float(np.quantile(dist_map[mask_bool], q_border))
    E = float(np.clip(q / d_max, 0.0, 1.0))

    # Sil
    Sil = compute_silhouette_score_v2(mask_bool)

    return np.array([A, C, E, Sil], dtype=float)


# =========================================================
# Prompt from sequence name
# =========================================================
def prompt_from_seq(seq: str) -> str:
    txt = seq.replace("-", " ").replace("_", " ").strip()
    if not txt:
        txt = "object"
    return f"{txt} ."


# =========================================================
# SAM2 + DINO inference for one image
# =========================================================
@torch.no_grad()
def run_sam2_dino_multimask(
    gdino,
    sam2_predictor,
    img_path: str,
    text_prompt: str,
    device: str,
    box_threshold: float,
    text_threshold: float,
):
    image_source, image = load_image(str(img_path))
    sam2_predictor.set_image(image_source)

    boxes, _, _ = predict(
        model=gdino,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    h, w, _ = image_source.shape

    if boxes is None or boxes.numel() == 0 or boxes.shape[0] == 0:
        return {
            "image_path": img_path,
            "img_height": int(h),
            "img_width": int(w),
            "text_prompt": text_prompt,
            "boxes_xyxy": [],
            "annotations": [],
            "raw_scores": [],
            "note": "no_box",
        }

    boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
    boxes_xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        masks, scores, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_xyxy,
            multimask_output=True,
        )

    if masks.ndim == 3:
        masks = masks[None, ...]
    if scores.ndim == 1:
        scores = scores[None, ...]

    B, M, Hm, Wm = masks.shape

    annotations = []
    raw_scores_flat = []
    idx = 0
    for b in range(B):
        for r in range(M):
            m = masks[b, r].astype(bool)
            rle = mask_to_rle(m)
            s = float(scores[b, r]) if scores is not None else 0.0

            annotations.append({
                "id": int(idx),
                "box_id": int(b),
                "rank_in_box": int(r),
                "segmentation": rle,
                "sam_score": s,
            })
            raw_scores_flat.append(s)
            idx += 1

    return {
        "image_path": img_path,
        "img_height": int(h),
        "img_width": int(w),
        "text_prompt": text_prompt,
        "boxes_xyxy": boxes_xyxy.tolist(),
        "annotations": annotations,
        "raw_scores": raw_scores_flat,
        "note": f"ok_B{B}_M{M}",
    }


# =========================================================
# Resume-aware saving
# =========================================================
def ensure_dirs(seq_out_dir: Path):
    (seq_out_dir / "json").mkdir(parents=True, exist_ok=True)
    (seq_out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (seq_out_dir / "gt_json").mkdir(parents=True, exist_ok=True)
    (seq_out_dir / "selected_by_score_optimized").mkdir(parents=True, exist_ok=True)
    (seq_out_dir / "selected_by_score_default_no_gt").mkdir(parents=True, exist_ok=True)
    (seq_out_dir / "selected_by_sam").mkdir(parents=True, exist_ok=True)


def frame_stem(frame_file: str) -> str:
    return Path(frame_file).stem


def result_json_path(seq_out_dir: Path, frame_file: str) -> Path:
    return seq_out_dir / "json" / f"{frame_stem(frame_file)}_result.json"


def gt_json_path(seq_out_dir: Path, frame_file: str) -> Path:
    return seq_out_dir / "gt_json" / f"{frame_stem(frame_file)}_gt.json"


def save_all_masks_png(seq_out_dir: Path, frame_file: str, annotations: List[Dict[str, Any]]):
    stem = frame_stem(frame_file)
    for ann in annotations:
        mid = ann["id"]
        m = rle_to_mask(ann["segmentation"])
        outp = seq_out_dir / "masks" / f"{stem}_m{mid:04d}.png"
        if outp.exists():
            continue
        cv2.imwrite(str(outp), (m.astype(np.uint8) * 255))


def build_and_save_gt_json(seq_out_dir: Path, davis_gt_file: str, img_path: str, frame_file: str):
    outp = gt_json_path(seq_out_dir, frame_file)
    if outp.exists():
        return
    gt_bool = load_davis_gt_bool(davis_gt_file)
    rle = mask_to_rle(gt_bool)
    h, w = gt_bool.shape[:2]
    obj = {
        "image_path": img_path,
        "gt_mask_path": davis_gt_file,
        "width": int(w),
        "height": int(h),
        "segmentation_rle": rle,
    }
    outp.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# =========================================================
# Optimizer helpers
# =========================================================
def load_gt_mask_from_gt_json(gt_json_path_: Path) -> Optional[np.ndarray]:
    with open(gt_json_path_, "r", encoding="utf-8") as f:
        gt = json.load(f)
    if "segmentation_rle" in gt and isinstance(gt["segmentation_rle"], dict):
        return rle_to_mask(gt["segmentation_rle"])
    return None


def decode_ann_mask_bool(ann) -> np.ndarray:
    return rle_to_mask(ann["segmentation"])


def load_seq_dataset(seq_out_dir: Path, max_files: Optional[int] = None):
    json_dir = seq_out_dir / "json"
    gt_dir = seq_out_dir / "gt_json"

    json_files = sorted(json_dir.glob("*_result.json"))
    if max_files:
        json_files = json_files[:max_files]

    dataset = []
    skipped_no_gt = 0
    skipped_bad = 0

    gt_index = {p.stem.replace("_gt", ""): p for p in gt_dir.glob("*_gt.json")}

    for fp in json_files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            skipped_bad += 1
            continue

        anns = data.get("annotations", [])
        if not anns:
            skipped_bad += 1
            continue

        key = fp.stem.replace("_result", "")
        gt_json = gt_index.get(key, None)
        if gt_json is None or not gt_json.exists():
            skipped_no_gt += 1
            continue

        gt_mask = load_gt_mask_from_gt_json(gt_json)
        if gt_mask is None:
            skipped_no_gt += 1
            continue

        H = int(data.get("img_height", gt_mask.shape[0]))
        W = int(data.get("img_width", gt_mask.shape[1]))
        if gt_mask.shape[0] != H or gt_mask.shape[1] != W:
            skipped_bad += 1
            continue

        masks = [decode_ann_mask_bool(a) for a in anns]

        dist_map = compute_distance_transform(H, W)
        d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

        raw_scores = data.get("raw_scores", [a.get("sam_score", a.get("score", 0.0)) for a in anns])

        dataset.append({
            "key": key,
            "W": W,
            "H": H,
            "masks": masks,
            "raw_scores": np.asarray(raw_scores, dtype=float),
            "gt": gt_mask,
            "dist_map": dist_map,
            "d_max": d_max,
        })

    dataset.sort(key=lambda x: x["key"])
    return dataset, skipped_no_gt, skipped_bad


def stratified_temporal_sample(dataset, ratio: float, seed: int, min_samples: int = 8):
    """
    Temporal bucket sampling:
      ratio=0.2 -> bucket_size=5 -> pick 1 frame from every 5 frames
      ratio=0.1 -> bucket_size=10 -> pick 1 frame from every 10 frames
    """
    if not dataset:
        return []

    ratio = float(np.clip(ratio, 1e-6, 1.0))
    bucket_size = max(1, int(round(1.0 / ratio)))

    rng = random.Random(seed)
    subset = []

    for start in range(0, len(dataset), bucket_size):
        bucket = dataset[start:start + bucket_size]
        if bucket:
            subset.append(rng.choice(bucket))

    if len(subset) < min_samples and len(dataset) > len(subset):
        chosen_keys = {item["key"] for item in subset}
        remaining = [item for item in dataset if item["key"] not in chosen_keys]
        rng.shuffle(remaining)
        need = min(min_samples - len(subset), len(remaining))
        subset.extend(remaining[:need])

    subset.sort(key=lambda x: x["key"])
    return subset


def eval_params(dataset, w_raw, q_border, t_area) -> float:
    w = normalize_weights(w_raw)
    q_border = float(np.clip(q_border, 0.0, 1.0))
    t_area = float(np.clip(t_area, 0.0, 1.0))

    ious = []
    for item in dataset:
        W, H = item["W"], item["H"]
        dist_map, d_max = item["dist_map"], item["d_max"]
        gt = item["gt"]
        masks = item["masks"]

        X = np.stack([mask_features(m, W, H, dist_map, d_max, q_border, t_area) for m in masks], axis=0)
        scores = X @ w
        top = int(np.argmax(scores))
        ious.append(compute_iou(masks[top], gt))

    return float(np.mean(ious)) if ious else 0.0


def optimize_random(train, val, seed=0, n_random=15000,
                    q_range=(0.05, 0.95), t_range=(0.05, 0.95)):
    rng = np.random.default_rng(seed)
    best = None
    best_val = -1.0

    for _ in range(n_random):
        w = rng.dirichlet(alpha=np.ones(4))
        q = rng.uniform(q_range[0], q_range[1])
        t = rng.uniform(t_range[0], t_range[1])
        v = eval_params(val, w, q, t)
        if v > best_val:
            best_val = v
            best = (w.copy(), float(q), float(t))

    w0, q0, t0 = best

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
        return -eval_params(val, w, q, t)

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

    res = minimize(objective, u0, method="Powell", options={"maxiter": 250, "disp": False})
    w_opt, q_opt, t_opt = pack(res.x)

    train_iou = eval_params(train, w_opt, q_opt, t_opt)
    val_iou = eval_params(val, w_opt, q_opt, t_opt)
    return w_opt, q_opt, t_opt, train_iou, val_iou


def optimize_for_sequence(seq_out_dir: Path, seed: int, gt_ratio: float,
                          n_random: int, q_low: float, q_high: float, t_low: float, t_high: float,
                          force_reopt: bool = False):
    """
    Optimization uses temporally stratified frame sampling.
    Baseline fixed params here are your CURRENT CODE baseline:
      w = [0.25, 0.25, 0.25, 0.25]
      q_border = 0.40
      t_area = 0.30
    """
    outp = seq_out_dir / "optimized_params.json"
    if outp.exists() and not force_reopt:
        return json.loads(outp.read_text(encoding="utf-8"))

    dataset, skipped_no_gt, skipped_bad = load_seq_dataset(seq_out_dir)
    if len(dataset) < 8:
        raise RuntimeError(
            f"[{seq_out_dir.name}] Too few usable frames with GT: {len(dataset)} "
            f"(skipped_no_gt={skipped_no_gt}, skipped_bad={skipped_bad})"
        )

    subset = stratified_temporal_sample(dataset, ratio=gt_ratio, seed=seed, min_samples=8)

    rnd = random.Random(seed)
    subset_for_split = subset.copy()
    rnd.shuffle(subset_for_split)

    n_train = max(1, int(len(subset_for_split) * 0.8))
    train = subset_for_split[:n_train]
    val = subset_for_split[n_train:] if len(subset_for_split) > 1 else subset_for_split

    q_range = (min(q_low, q_high), max(q_low, q_high))
    t_range = (min(t_low, t_high), max(t_low, t_high))

    w_eq = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    q0 = float(np.clip(0.40, *q_range))
    t0 = float(np.clip(0.30, *t_range))
    base_train = eval_params(train, w_eq, q0, t0)
    base_val = eval_params(val, w_eq, q0, t0) if len(val) else base_train

    w_opt, q_opt, t_opt, train_iou, val_iou = optimize_random(
        train,
        val if len(val) else train,
        seed=seed,
        n_random=n_random,
        q_range=q_range,
        t_range=t_range,
    )

    obj = {
        "sequence": seq_out_dir.name,
        "n_total_frames_with_gt": len(dataset),
        "n_used_for_opt": len(subset),
        "opt_seed": seed,
        "gt_ratio": gt_ratio,
        "sampling_mode": "temporal_bucket_random_one_per_bucket",
        "baseline_for_optimization": {
            "w": w_eq.tolist(),
            "q_border": q0,
            "t_area": t0,
            "train_meanIoU": float(base_train),
            "val_meanIoU": float(base_val),
        },
        "optimized": {
            "w": w_opt.tolist(),
            "q_border": float(q_opt),
            "t_area": float(t_opt),
            "train_meanIoU": float(train_iou),
            "val_meanIoU": float(val_iou),
        },
    }
    outp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return obj


# =========================================================
# Evaluation
#   baseline in evaluation = argmax(raw SAM confidence)
#   score-based = argmax(features @ w)
# =========================================================
def evaluate_sequence(seq_out_dir: Path,
                      mode: str = "optimized",
                      custom_params: Optional[Dict[str, Any]] = None,
                      force_reeval: bool = False):
    """
    mode:
      - optimized
      - default_no_gt

    For default_no_gt, we use the CURRENT CODE fixed baseline params:
      w = [0.25, 0.25, 0.25, 0.25]
      q_border = 0.40
      t_area = 0.30
    """
    if mode == "optimized":
        outp = seq_out_dir / "eval_summary_optimized.json"
        score_dir = seq_out_dir / "selected_by_score_optimized"
    elif mode == "default_no_gt":
        outp = seq_out_dir / "eval_summary_default_no_gt.json"
        score_dir = seq_out_dir / "selected_by_score_default_no_gt"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if outp.exists() and not force_reeval:
        return json.loads(outp.read_text(encoding="utf-8"))

    if mode == "optimized":
        param_path = seq_out_dir / "optimized_params.json"
        if not param_path.exists():
            raise RuntimeError(f"Missing optimized_params.json in {seq_out_dir}")
        params = json.loads(param_path.read_text(encoding="utf-8"))
        w = np.asarray(params["optimized"]["w"], dtype=float)
        q_border = float(params["optimized"]["q_border"])
        t_area = float(params["optimized"]["t_area"])
    else:
        if custom_params is None:
            custom_params = {
                "w": [0.25, 0.25, 0.25, 0.25],
                "q_border": 0.40,
                "t_area": 0.30,
            }
        w = np.asarray(custom_params["w"], dtype=float)
        q_border = float(custom_params["q_border"])
        t_area = float(custom_params["t_area"])

    w = normalize_weights(w)

    dataset, skipped_no_gt, skipped_bad = load_seq_dataset(seq_out_dir)

    baseline_ious = []
    scored_ious = []
    per_frame_records = []

    best_improve = {"key": None, "delta": -1e9}
    worst_drop = {"key": None, "delta": 1e9}

    sam_dir = seq_out_dir / "selected_by_sam"
    sam_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    for item in tqdm(dataset, desc=f"Eval {seq_out_dir.name} [{mode}]", leave=False):
        key = item["key"]
        W, H = item["W"], item["H"]
        dist_map, d_max = item["dist_map"], item["d_max"]
        gt = item["gt"]
        masks = item["masks"]
        raw_scores = item["raw_scores"]

        # evaluation baseline = raw SAM best
        bidx = int(np.argmax(raw_scores))
        biou = compute_iou(masks[bidx], gt)

        sam_png = sam_dir / f"{key}_sam_best.png"
        if not sam_png.exists():
            cv2.imwrite(str(sam_png), (masks[bidx].astype(np.uint8) * 255))

        # score-based
        X = np.stack([mask_features(m, W, H, dist_map, d_max, q_border, t_area) for m in masks], axis=0)
        s_scores = X @ w
        sidx = int(np.argmax(s_scores))
        siou = compute_iou(masks[sidx], gt)

        sel_png = score_dir / f"{key}_best.png"
        if not sel_png.exists():
            cv2.imwrite(str(sel_png), (masks[sidx].astype(np.uint8) * 255))

        delta = siou - biou
        if delta > best_improve["delta"]:
            best_improve = {
                "key": key,
                "delta": float(delta),
                "baseline_iou": float(biou),
                "scored_iou": float(siou),
            }
        if delta < worst_drop["delta"]:
            worst_drop = {
                "key": key,
                "delta": float(delta),
                "baseline_iou": float(biou),
                "scored_iou": float(siou),
            }

        baseline_ious.append(biou)
        scored_ious.append(siou)
        per_frame_records.append({
            "frame": key,
            "mode": mode,
            "baseline_best_idx": bidx,
            "scored_best_idx": sidx,
            "baseline_iou": float(biou),
            "scored_iou": float(siou),
            "delta": float(delta),
            "sam_best_mask_png": str(sam_png),
            "score_best_mask_png": str(sel_png),
        })

    def stats(arr):
        arr = np.asarray(arr, dtype=float)
        return {
            "mean": float(arr.mean()) if arr.size else None,
            "min": float(arr.min()) if arr.size else None,
            "max": float(arr.max()) if arr.size else None,
            "n": int(arr.size),
        }

    summary = {
        "sequence": seq_out_dir.name,
        "mode": mode,
        "params": {
            "w": w.tolist(),
            "q_border": float(q_border),
            "t_area": float(t_area),
        },
        "n_frames_evaluated": len(per_frame_records),
        "baseline_iou": stats(baseline_ious),
        "scored_iou": stats(scored_ious),
        "best_improvement": best_improve,
        "worst_drop": worst_drop,
        "per_frame": per_frame_records,
    }

    outp.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csvp = seq_out_dir / f"eval_per_frame_{mode}.csv"
    if not csvp.exists() or force_reeval:
        import pandas as pd
        pd.DataFrame(per_frame_records).to_csv(csvp, index=False)

    return summary


# =========================================================
# Step1 runner per sequence
# =========================================================
def process_sequence(
    davis_root: str,
    resolution: str,
    seq: str,
    seq_out_dir: Path,
    gdino,
    sam2_predictor,
    device: str,
    box_threshold: float,
    text_threshold: float,
    force_prompt: Optional[str] = None,
):
    ensure_dirs(seq_out_dir)

    frames = list_frames(davis_root, resolution, seq)
    if len(frames) == 0:
        return

    prompt = force_prompt if force_prompt else prompt_from_seq(seq)

    for ff in tqdm(frames, desc=f"SAM2+DINO {seq}", leave=False):
        out_json = result_json_path(seq_out_dir, ff)
        if out_json.exists():
            imgp = davis_img_path(davis_root, resolution, seq, ff)
            gtp = davis_gt_path(davis_root, resolution, seq, ff)
            if gtp:
                build_and_save_gt_json(seq_out_dir, gtp, imgp, ff)
            continue

        imgp = davis_img_path(davis_root, resolution, seq, ff)
        gtp = davis_gt_path(davis_root, resolution, seq, ff)

        data = run_sam2_dino_multimask(
            gdino=gdino,
            sam2_predictor=sam2_predictor,
            img_path=imgp,
            text_prompt=prompt,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

        anns = data.get("annotations", [])
        if anns:
            save_all_masks_png(seq_out_dir, ff, anns)

        if gtp:
            build_and_save_gt_json(seq_out_dir, gtp, imgp, ff)



def evaluate_default_ablations(seq_out_dir: Path, force_reeval: bool = False):
    """
    Run ablations on the fixed default_no_gt params:
      default = w=[0.25,0.25,0.25,0.25], q_border=0.40, t_area=0.30

    Ablation rule:
      set one weight to 0, then normalize the remaining weights.

    Outputs:
      - ablation_summary_default_no_gt.json
      - ablation_per_frame_default_no_gt.csv
      - selected_by_score_ablation_<name>/
    """
    outp = seq_out_dir / "ablation_summary_default_no_gt.json"
    csvp = seq_out_dir / "ablation_per_frame_default_no_gt.csv"

    if outp.exists() and csvp.exists() and not force_reeval:
        return json.loads(outp.read_text(encoding="utf-8"))

    dataset, skipped_no_gt, skipped_bad = load_seq_dataset(seq_out_dir)

    base_params = {
        "w": np.array([0.15, 0.20, 0.40, 0.25], dtype=float),
        "q_border": 0.30,
        "t_area": 0.25,
    }

    ablations = {
        "default_full": np.array([0.25, 0.25, 0.25, 0.25], dtype=float),
        "no_area": np.array([0.00, 0.25, 0.25, 0.25], dtype=float),
        "no_center": np.array([0.25, 0.00, 0.25, 0.25], dtype=float),
        "no_border": np.array([0.25, 0.25, 0.00, 0.25], dtype=float),
        "no_silhouette": np.array([0.25, 0.25, 0.25, 0.00], dtype=float),
    }

    per_frame_records = []
    ablation_stats = {}

    sam_dir = seq_out_dir / "selected_by_sam"
    sam_dir.mkdir(parents=True, exist_ok=True)

    for ablation_name, w_raw in ablations.items():
        w = normalize_weights(w_raw)
        q_border = base_params["q_border"]
        t_area = base_params["t_area"]

        score_dir = seq_out_dir / f"selected_by_score_ablation_{ablation_name}"
        score_dir.mkdir(parents=True, exist_ok=True)

        baseline_ious = []
        scored_ious = []
        best_improve = {"key": None, "delta": -1e9}
        worst_drop = {"key": None, "delta": 1e9}

        for item in tqdm(dataset, desc=f"Ablation {seq_out_dir.name} [{ablation_name}]", leave=False):
            key = item["key"]
            W, H = item["W"], item["H"]
            dist_map, d_max = item["dist_map"], item["d_max"]
            gt = item["gt"]
            masks = item["masks"]
            raw_scores = item["raw_scores"]

            # baseline = raw SAM best
            bidx = int(np.argmax(raw_scores))
            biou = compute_iou(masks[bidx], gt)

            sam_png = sam_dir / f"{key}_sam_best.png"
            if not sam_png.exists():
                cv2.imwrite(str(sam_png), (masks[bidx].astype(np.uint8) * 255))

            # score-based with ablated weights
            X = np.stack(
                [mask_features(m, W, H, dist_map, d_max, q_border, t_area) for m in masks],
                axis=0
            )
            s_scores = X @ w
            sidx = int(np.argmax(s_scores))
            siou = compute_iou(masks[sidx], gt)

            sel_png = score_dir / f"{key}_best.png"
            if not sel_png.exists():
                cv2.imwrite(str(sel_png), (masks[sidx].astype(np.uint8) * 255))

            delta = siou - biou
            if delta > best_improve["delta"]:
                best_improve = {
                    "key": key,
                    "delta": float(delta),
                    "baseline_iou": float(biou),
                    "scored_iou": float(siou),
                }
            if delta < worst_drop["delta"]:
                worst_drop = {
                    "key": key,
                    "delta": float(delta),
                    "baseline_iou": float(biou),
                    "scored_iou": float(siou),
                }

            baseline_ious.append(biou)
            scored_ious.append(siou)

            per_frame_records.append({
                "sequence": seq_out_dir.name,
                "ablation": ablation_name,
                "frame": key,
                "baseline_best_idx": bidx,
                "scored_best_idx": sidx,
                "baseline_iou": float(biou),
                "scored_iou": float(siou),
                "delta": float(delta),
                "weights_after_normalize": w.tolist(),
                "q_border": float(q_border),
                "t_area": float(t_area),
                "sam_best_mask_png": str(sam_png),
                "score_best_mask_png": str(sel_png),
            })

        baseline_arr = np.asarray(baseline_ious, dtype=float)
        scored_arr = np.asarray(scored_ious, dtype=float)

        ablation_stats[ablation_name] = {
            "weights_after_normalize": w.tolist(),
            "q_border": float(q_border),
            "t_area": float(t_area),
            "n_frames_evaluated": int(len(scored_arr)),
            "baseline_iou": {
                "mean": float(baseline_arr.mean()) if baseline_arr.size else None,
                "min": float(baseline_arr.min()) if baseline_arr.size else None,
                "max": float(baseline_arr.max()) if baseline_arr.size else None,
                "n": int(baseline_arr.size),
            },
            "scored_iou": {
                "mean": float(scored_arr.mean()) if scored_arr.size else None,
                "min": float(scored_arr.min()) if scored_arr.size else None,
                "max": float(scored_arr.max()) if scored_arr.size else None,
                "n": int(scored_arr.size),
            },
            "mean_delta_vs_sam_baseline": float((scored_arr - baseline_arr).mean()) if scored_arr.size else None,
            "best_improvement": best_improve,
            "worst_drop": worst_drop,
        }

    summary = {
        "sequence": seq_out_dir.name,
        "base_setting": {
            "w_before_ablation": base_params["w"].tolist(),
            "q_border": float(base_params["q_border"]),
            "t_area": float(base_params["t_area"]),
        },
        "ablations": ablation_stats,
    }

    outp.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import pandas as pd
    pd.DataFrame(per_frame_records).to_csv(csvp, index=False)

    return summary
# =========================================================
# Main orchestration
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis_root", type=str,
                    default=r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS")
    ap.add_argument("--resolution", type=str, default="480p", choices=["480p", "1080p"])
    ap.add_argument("--out_root", type=str,
                    default=r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results")

    # models
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sam_ckpt", type=str, default="./checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--sam_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--dino_cfg", type=str,
                    default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    ap.add_argument("--dino_ckpt", type=str,
                    default="gdino_checkpoints/groundingdino_swint_ogc.pth")
    ap.add_argument("--box_th", type=float, default=0.20)
    ap.add_argument("--text_th", type=float, default=0.20)

    # optimization
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gt_ratio", type=float, default=0.20)
    ap.add_argument("--n_random", type=int, default=1000)
    ap.add_argument("--q_low", type=float, default=0.05)
    ap.add_argument("--q_high", type=float, default=0.95)
    ap.add_argument("--t_low", type=float, default=0.05)
    ap.add_argument("--t_high", type=float, default=0.95)

    # run control
    ap.add_argument("--only_seq", type=str, default="", help="if set, only run this sequence, e.g. bear")
    ap.add_argument("--skip_infer", action="store_true",
                    help="reuse existing *_result.json and do not rerun DINO+SAM2")
    ap.add_argument("--skip_opt", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--force_reopt", action="store_true",
                    help="ignore existing optimized_params.json and recompute")
    ap.add_argument("--force_reeval", action="store_true",
                    help="ignore existing eval summaries and recompute")
    ap.add_argument("--run_ablation", action="store_true",
                    help="run default_no_gt weight ablations")
    ap.add_argument("--skip_ablation", action="store_true",
                    help="skip default_no_gt ablations even if enabled elsewhere")
    args = ap.parse_args()

    davis_root = find_davis_root(args.davis_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    gdino = None
    sam2_predictor = None

    if not args.skip_infer:
        print("Loading SAM2...")
        sam2_model = build_sam2(args.sam_cfg, args.sam_ckpt, device=args.device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)

        print("Loading GroundingDINO...")
        gdino = load_model(
            model_config_path=args.dino_cfg,
            model_checkpoint_path=args.dino_ckpt,
            device=args.device,
        )

        torch.set_float32_matmul_precision("high")
    else:
        print("[INFO] skip_infer=True, reusing existing json/masks/gt_json")

    seqs = list_sequences(davis_root, args.resolution)
    if args.only_seq:
        if args.only_seq not in seqs:
            raise SystemExit(f"Sequence not found in DAVIS/{args.resolution}: {args.only_seq}")
        seqs = [args.only_seq]

    all_summaries = []
    all_ablation_summaries = []
    for seq in tqdm(seqs, desc="DAVIS sequences"):
        seq_out_dir = out_root / seq
        seq_out_dir.mkdir(parents=True, exist_ok=True)
        ensure_dirs(seq_out_dir)

        # Step1: inference
        if not args.skip_infer:
            process_sequence(
                davis_root=davis_root,
                resolution=args.resolution,
                seq=seq,
                seq_out_dir=seq_out_dir,
                gdino=gdino,
                sam2_predictor=sam2_predictor,
                device=args.device,
                box_threshold=args.box_th,
                text_threshold=args.text_th,
            )

        # Step2: optimize params per sequence
        if not args.skip_opt:
            _ = optimize_for_sequence(
                seq_out_dir=seq_out_dir,
                seed=args.seed,
                gt_ratio=args.gt_ratio,
                n_random=args.n_random,
                q_low=args.q_low,
                q_high=args.q_high,
                t_low=args.t_low,
                t_high=args.t_high,
                force_reopt=args.force_reopt,
            )

        # Step3A: optimized eval
        if not args.skip_eval:
            summ_opt = evaluate_sequence(
                seq_out_dir,
                mode="optimized",
                force_reeval=args.force_reeval,
            )
            all_summaries.append(summ_opt)

            # Step3B: no-GT-search, fixed current-code baseline params
            summ_default = evaluate_sequence(
                seq_out_dir,
                mode="default_no_gt",
                custom_params={
                    "w": [0.15, 0.20, 0.40, 0.25],
                    "q_border": 0.30,
                    "t_area": 0.25,
                },
                force_reeval=args.force_reeval,
            )
            all_summaries.append(summ_default)

            # Step3C: ablations on default_no_gt
            if args.run_ablation and not args.skip_ablation:
                ablation_summary = evaluate_default_ablations(
                    seq_out_dir,
                    force_reeval=args.force_reeval,
                )
                all_ablation_summaries.append(ablation_summary)

    global_path = out_root / f"global_summary_{args.resolution}.json"
    if all_summaries:
        global_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
        print(f"[DONE] Global summary saved to: {global_path}")
    print("[DONE] All sequences processed.")
    if all_ablation_summaries:
        ablation_global_path = out_root / f"global_ablation_summary_{args.resolution}.json"
        ablation_global_path.write_text(json.dumps(all_ablation_summaries, indent=2), encoding="utf-8")
        print(f"[DONE] Global ablation summary saved to: {ablation_global_path}")


if __name__ == "__main__":
    main()

