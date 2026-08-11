import os
import json
import csv
import time
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image  # 确保有这行

import numpy as np
import cv2
from tqdm import tqdm

import torch
from torchvision.datasets import VOCSegmentation
from torchvision.ops import box_convert

from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# =========================
# VOC classes
# =========================
VOC_CLASSES = {
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor",
}


def find_voc2012_root(voc_root: str) -> Path:
    """
    Accept either:
      - .../VOC2012_train_val
      - .../VOC2012_train_val/VOCdevkit/VOC2012
      - .../VOCdevkit/VOC2012
    Return the Path to .../VOCdevkit/VOC2012
    """
    p = Path(voc_root)

    # already points to VOC2012
    cand = p / "JPEGImages"
    if cand.exists():
        return p

    # points to VOCdevkit
    cand = p / "VOC2012" / "JPEGImages"
    if cand.exists():
        return p / "VOC2012"

    # points to archive root that contains VOCdevkit/VOC2012
    cand = p / "VOCdevkit" / "VOC2012" / "JPEGImages"
    if cand.exists():
        return p / "VOCdevkit" / "VOC2012"

    raise FileNotFoundError(
        f"Cannot locate VOC2012 under: {voc_root}\n"
        f"Expected one of:\n"
        f"  <root>/VOCdevkit/VOC2012/JPEGImages\n"
        f"  <root>/VOC2012/JPEGImages\n"
        f"  <root>/JPEGImages\n"
    )

def load_split_ids(voc2012_root: Path, split: str) -> List[str]:
    split_file = voc2012_root / "ImageSets" / "Segmentation" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids

def read_voc_by_id(voc2012_root: Path, img_id: str) -> Tuple[str, str, np.ndarray, np.ndarray]:
    img_path = voc2012_root / "JPEGImages" / f"{img_id}.jpg"
    seg_path = voc2012_root / "SegmentationClass" / f"{img_id}.png"

    if not img_path.exists():
        raise FileNotFoundError(f"Missing image: {img_path}")
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing seg: {seg_path}")

    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ✅ 关键：用 PIL 读 palette PNG，保留类别 id
    seg_mask = np.array(Image.open(seg_path), dtype=np.uint8)

    return str(img_path), str(seg_path), image_rgb, seg_mask
def build_dino_prompt(class_name: str) -> str:
    special = {
        "diningtable": "a photo of a dining table.",
        "pottedplant": "a photo of a potted plant.",
        "tvmonitor": "a photo of a tv monitor.",
        "motorbike": "a photo of a motorbike.",
        "aeroplane": "a photo of an aeroplane.",
    }
    if class_name in special:
        return special[class_name]
    return f"a photo of a {class_name}."


# =========================
# Metrics / stats
# =========================
def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 0.0


def summarize(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(len(arr)),
    }


# =========================
# Score function (same core as你现在版本)
# =========================
def compute_distance_transform(h, w):
    border_mask = np.zeros((h, w), np.uint8)
    border_mask[1:-1, 1:-1] = 1
    return cv2.distanceTransform(border_mask, distanceType=cv2.DIST_L2, maskSize=5)


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


def normalize_w(w: np.ndarray) -> np.ndarray:
    w = np.array(w, dtype=float)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 1e-12:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    return w / s


# =========================
# Cache for fast search
# =========================
@dataclass
class CandidateCache:
    area_ratio: np.ndarray         # (K,)
    center: np.ndarray             # (K,)
    sil: np.ndarray                # (K,)
    iou: np.ndarray                # (K,) IoU vs GT(class) for each candidate mask
    dist_sorted: List[np.ndarray]  # list[K] sorted dist_map values within mask
    d_max: float
    W: int
    H: int


def compute_E_from_sorted(dist_sorted_list: List[np.ndarray], q_border: float, d_max: float) -> np.ndarray:
    q = float(np.clip(q_border, 0.0, 1.0))
    out = np.zeros((len(dist_sorted_list),), dtype=np.float32)
    inv = 1.0 / (d_max + 1e-12)
    for i, arr in enumerate(dist_sorted_list):
        n = int(arr.shape[0])
        if n <= 0:
            out[i] = 0.0
        else:
            idx = int(round(q * (n - 1)))
            out[i] = float(np.clip(float(arr[idx]) * inv, 0.0, 1.0))
    return out


def select_iou_by_score(cache: CandidateCache, w, q_border, t_area) -> float:
    w = normalize_w(w)
    wA, wC, wB, wS = [float(x) for x in w]

    A = area_term_parabola_vec(cache.area_ratio, float(t_area))
    C = cache.center
    S = cache.sil
    E = compute_E_from_sorted(cache.dist_sorted, float(q_border), cache.d_max)

    scores = wA * A + wC * C + wB * E + wS * S
    top = int(np.argmax(scores))
    return float(cache.iou[top])


def eval_params_fast(caches: List[CandidateCache], w, q_border, t_area) -> float:
    if len(caches) == 0:
        return 0.0
    vals = [select_iou_by_score(c, w, q_border, t_area) for c in caches]
    return float(np.mean(vals))


def optimize_random(caches_train: List[CandidateCache],
                    caches_val: List[CandidateCache],
                    seed: int,
                    n_random: int,
                    q_range=(0.05, 0.95),
                    t_range=(0.05, 0.95)):
    rng = np.random.default_rng(seed)
    best_val = -1.0
    best = None

    for _ in tqdm(range(n_random), desc="Search", unit="trial"):
        w = rng.dirichlet(np.ones(4))
        q = float(rng.uniform(*q_range))
        t = float(rng.uniform(*t_range))
        v = eval_params_fast(caches_val, w, q, t)
        if v > best_val:
            best_val = v
            best = (w.copy(), q, t)

    w0, q0, t0 = best

    # local refinement (Powell) on val
    from scipy.optimize import minimize

    def softplus(x):
        return np.log1p(np.exp(x))

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def inv_softplus(y):
        y = np.clip(y, 1e-6, None)
        return np.log(np.exp(y) - 1.0 + 1e-12)

    def inv_sigmoid(y):
        y = np.clip(y, 1e-6, 1 - 1e-6)
        return np.log(y / (1 - y))

    def pack(u):
        w_pos = softplus(u[:4])
        w = w_pos / (w_pos.sum() + 1e-12)
        q = q_range[0] + (q_range[1] - q_range[0]) * sigmoid(u[4])
        t = t_range[0] + (t_range[1] - t_range[0]) * sigmoid(u[5])
        return w, float(q), float(t)

    def objective(u):
        w, q, t = pack(u)
        return -eval_params_fast(caches_val, w, q, t)

    u_w = inv_softplus(w0 + 1e-6)
    qn = (q0 - q_range[0]) / (q_range[1] - q_range[0] + 1e-12)
    tn = (t0 - t_range[0]) / (t_range[1] - t_range[0] + 1e-12)
    u0 = np.concatenate([u_w, [inv_sigmoid(qn), inv_sigmoid(tn)]]).astype(float)

    res = minimize(objective, u0, method="Powell", options={"maxiter": 120, "disp": False})
    w_opt, q_opt, t_opt = pack(res.x)

    train_iou = eval_params_fast(caches_train, w_opt, q_opt, t_opt)
    val_iou = eval_params_fast(caches_val, w_opt, q_opt, t_opt)
    return w_opt, q_opt, t_opt, train_iou, val_iou


# =========================
# VOC utilities
# =========================
def voc_gt_for_class(seg_mask: np.ndarray, class_id: int) -> np.ndarray:
    # VOC: 0 bg, 1..20 class, 255 void
    return (seg_mask == class_id)


def read_voc_item(voc: VOCSegmentation, idx: int) -> Tuple[np.ndarray, np.ndarray]:
    img, target = voc[idx]
    img = np.array(img.convert("RGB"))
    target = np.array(target, dtype=np.uint8)
    return img, target


def sample_indices_per_class(voc: VOCSegmentation, class_id: int, sample_ratio: float, seed: int, max_per_class: int = 0) -> List[int]:
    rng = random.Random(seed + class_id * 997)
    indices = []
    for i in tqdm(range(len(voc)), desc=f"Scan {VOC_CLASSES[class_id]}", unit="img"):
        _, t = voc[i]
        m = np.array(t, dtype=np.uint8)
        if (m == class_id).any():
            indices.append(i)

    if not indices:
        return []
    rng.shuffle(indices)
    k = max(1, int(len(indices) * sample_ratio))
    if max_per_class > 0:
        k = min(k, max_per_class)
    return indices[:k]


# =========================
# Model loading (SAM2 + DINO) using YOUR snippet
# =========================
def load_models(args, device: str):
    torch.set_float32_matmul_precision("high")

    print("Loading SAM2...")
    sam2_model = build_sam2(args.sam2_model_config, args.sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    print("Loading GroundingDINO...")
    gdino = load_model(
        model_config_path=args.gdino_config,
        model_checkpoint_path=args.gdino_checkpoint,
        device=device,
    )

    return gdino, sam2_predictor


@torch.no_grad()
def infer_dino_boxes(image_path: str, gdino, caption: str, device: str, box_thresh: float, text_thresh: float) -> Optional[np.ndarray]:
    """
    returns boxes_xyxy in pixel coords: (N,4) float32
    """
    image_source, image = load_image(image_path)
    boxes, _, _ = predict(
        model=gdino,
        image=image,
        caption=caption,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
        device=device,
    )
    if boxes is None or boxes.shape[0] == 0:
        return None

    h, w, _ = image_source.shape
    boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
    boxes_xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy().astype(np.float32)
    return boxes_xyxy


@torch.no_grad()
def infer_sam2_masks(image_source_rgb: np.ndarray, boxes_xyxy: np.ndarray, sam2_predictor, device: str, multimask_output: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    boxes_xyxy: (B,4)
    returns:
      masks_flat: (K,H,W) bool
      scores_flat: (K,) float
    """
    sam2_predictor.set_image(image_source_rgb)

    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.startswith("cuda") else torch.cpu.amp.autocast(enabled=False)
    with ctx:
        masks, scores, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_xyxy,
            multimask_output=multimask_output,
        )

    # masks: (B,3,H,W) if multimask, else (B,1,H,W) or (B,H,W) depending on impl
    masks = np.array(masks)
    scores = np.array(scores).reshape(-1)

    if masks.ndim == 4:
        # (B,M,H,W) -> flatten
        masks_flat = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
    elif masks.ndim == 3:
        masks_flat = masks
    else:
        raise RuntimeError(f"Unexpected masks shape: {masks.shape}")

    masks_flat = masks_flat.astype(bool)
    return masks_flat, scores.astype(float)


# =========================
# Build cache for one (image, class)
# =========================
def build_cache_for_image_class(
    voc_root_img_path: str,
    image_rgb: np.ndarray,
    seg_mask: np.ndarray,
    class_id: int,
    gdino,
    sam2_predictor,
    device: str,
    box_thresh: float,
    text_thresh: float,
    multimask: bool,
    dist_dtype: str = "float16",
) -> Optional[Dict]:
    gt = voc_gt_for_class(seg_mask, class_id)
    if gt.sum() == 0:
        return None

    prompt = build_dino_prompt(VOC_CLASSES[class_id])

    boxes_xyxy = infer_dino_boxes(
        image_path=voc_root_img_path,
        gdino=gdino,
        caption=prompt,
        device=device,
        box_thresh=box_thresh,
        text_thresh=text_thresh,
    )
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return None

    H, W = seg_mask.shape
    dist_map = compute_distance_transform(H, W)
    d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

    # SAM2 masks for ALL boxes
    masks_flat, sam_scores = infer_sam2_masks(
        image_source_rgb=image_rgb,
        boxes_xyxy=boxes_xyxy,
        sam2_predictor=sam2_predictor,
        device=device,
        multimask_output=multimask,
    )
    if masks_flat is None or len(masks_flat) == 0:
        return None

    # baseline = SAM2 top1 by SAM score among all candidates
    best_idx = int(np.argmax(sam_scores))
    baseline_iou = compute_iou(masks_flat[best_idx], gt)

    # precompute features per candidate
    img_area = float(H * W)
    cx, cy = W / 2.0, H / 2.0
    norm = (0.5 * np.hypot(W, H)) + 1e-12

    K = masks_flat.shape[0]
    area_ratio = np.zeros((K,), dtype=np.float32)
    center = np.zeros((K,), dtype=np.float32)
    sil = np.zeros((K,), dtype=np.float32)
    iou = np.zeros((K,), dtype=np.float32)
    dist_sorted: List[np.ndarray] = [None] * K

    for j in range(K):
        m = masks_flat[j]
        a = float(m.sum())
        if a <= 0:
            area_ratio[j] = 0.0
            center[j] = 0.0
            sil[j] = 0.0
            iou[j] = 0.0
            dist_sorted[j] = np.zeros((1,), dtype=np.float16 if dist_dtype == "float16" else np.float32)
            continue

        area_ratio[j] = a / img_area

        ys, xs = np.where(m)
        mx, my = float(xs.mean()), float(ys.mean())
        dp = np.hypot(mx - cx, my - cy) / norm
        center[j] = 1.0 - float(np.clip(dp, 0.0, 1.0))

        sil[j] = compute_silhouette_score_v2(m)
        iou[j] = compute_iou(m, gt)

        vals = dist_map[m].astype(np.float32, copy=False)
        vals.sort()
        if dist_dtype == "float16":
            vals = vals.astype(np.float16, copy=False)
        dist_sorted[j] = vals

    cache = CandidateCache(
        area_ratio=area_ratio,
        center=center,
        sil=sil,
        iou=iou,
        dist_sorted=dist_sorted,
        d_max=d_max,
        W=W,
        H=H,
    )
    return {"cache": cache, "baseline_iou": float(baseline_iou), "prompt": prompt, "K": int(K)}


# =========================
# Experiment runner
# =========================
def split_train_val(caches: List[CandidateCache], split: float, seed: int):
    rnd = random.Random(seed)
    arr = caches[:]
    rnd.shuffle(arr)
    n_train = max(1, int(len(arr) * split))
    return arr[:n_train], arr[n_train:]


def run(args):
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Device: {device}")

    # -------------------------
    # Resolve VOC root + load split ids
    # -------------------------
    voc2012_root = find_voc2012_root(args.voc_root)
    print(f"VOC2012 root resolved to: {voc2012_root}")

    img_ids = load_split_ids(voc2012_root, args.voc_split)
    print(f"Split '{args.voc_split}': {len(img_ids)} images")

    # -------------------------
    # Output folders
    # -------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"voc{args.voc_year}_{args.voc_split}_ratio{args.sample_ratio}_{ts}"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args) | {"device_used": device, "voc2012_root": str(voc2012_root)}, f, indent=2)

    # cache directory
    cache_dir = run_dir / "cache_npz"
    cache_dir.mkdir(exist_ok=True)

    # -------------------------
    # 1) Sample ids per class by scanning GT masks
    # -------------------------
    sampled_ids: Dict[int, List[str]] = {cid: [] for cid in range(1, 21)}
    for cid in range(1, 21):
        cname = VOC_CLASSES[cid]
        hits: List[str] = []

        for img_id in tqdm(img_ids, desc=f"Scan {cname}", unit="img"):
            seg_path = voc2012_root / "SegmentationClass" / f"{img_id}.png"
            seg = np.array(Image.open(seg_path), dtype=np.uint8)  # ✅ PIL
            if (seg == cid).any():
                hits.append(img_id)

        if not hits:
            sampled_ids[cid] = []
            continue

        rng = random.Random(args.seed + cid * 997)
        rng.shuffle(hits)

        k = max(1, int(len(hits) * args.sample_ratio))
        if args.max_per_class > 0:
            k = min(k, args.max_per_class)

        sampled_ids[cid] = hits[:k]
        print(f"Class {cname}: found={len(hits)}, sampled={len(sampled_ids[cid])}")

    with open(run_dir / "sampled_ids.json", "w", encoding="utf-8") as f:
        json.dump({VOC_CLASSES[c]: sampled_ids[c] for c in sampled_ids}, f, indent=2)

    # -------------------------
    # 2) Load models once
    # -------------------------
    gdino, sam2_predictor = load_models(args, device)

    # -------------------------
    # 3) Build per-class caches (heavy: DINO+SAM2)
    # -------------------------
    per_class_caches: Dict[int, List[CandidateCache]] = {cid: [] for cid in range(1, 21)}
    per_class_baseline: Dict[int, List[float]] = {cid: [] for cid in range(1, 21)}

    def cache_path(cid: int, img_id: str) -> Path:
        return cache_dir / f"{VOC_CLASSES[cid]}__{img_id}.npz"

    for cid in range(1, 21):
        cname = VOC_CLASSES[cid]
        ids = sampled_ids[cid]
        if not ids:
            continue

        for img_id in tqdm(ids, desc=f"Infer {cname}", unit="img"):
            p = cache_path(cid, img_id)

            if p.exists() and args.use_disk_cache:
                data = np.load(p, allow_pickle=True)
                cache = CandidateCache(
                    area_ratio=data["area_ratio"],
                    center=data["center"],
                    sil=data["sil"],
                    iou=data["iou"],
                    dist_sorted=list(data["dist_sorted"]),
                    d_max=float(data["d_max"]),
                    W=int(data["W"]),
                    H=int(data["H"]),
                )
                per_class_caches[cid].append(cache)
                per_class_baseline[cid].append(float(data["baseline_iou"]))
                continue

            # read actual image + seg
            img_path, seg_path, image_rgb, seg = read_voc_by_id(voc2012_root, img_id)

            pack = build_cache_for_image_class(
                voc_root_img_path=img_path,     # DINO uses this path via load_image
                image_rgb=image_rgb,
                seg_mask=seg,
                class_id=cid,
                gdino=gdino,
                sam2_predictor=sam2_predictor,
                device=device,
                box_thresh=args.dino_box_thresh,
                text_thresh=args.dino_text_thresh,
                multimask=args.sam_multimask,
                dist_dtype=args.dist_dtype,
            )
            if pack is None:
                continue

            cache = pack["cache"]
            baseline_iou = float(pack["baseline_iou"])

            per_class_caches[cid].append(cache)
            per_class_baseline[cid].append(baseline_iou)

            if args.use_disk_cache:
                np.savez_compressed(
                    p,
                    area_ratio=cache.area_ratio,
                    center=cache.center,
                    sil=cache.sil,
                    iou=cache.iou,
                    dist_sorted=np.array(cache.dist_sorted, dtype=object),
                    d_max=np.array(cache.d_max),
                    W=np.array(cache.W),
                    H=np.array(cache.H),
                    baseline_iou=np.array(baseline_iou),
                )

    # -------------------------
    # 4) Build "overall" set (weighted by number of sampled instances)
    # -------------------------
    overall_caches: List[CandidateCache] = []
    overall_baseline: List[float] = []
    for cid in range(1, 21):
        overall_caches.extend(per_class_caches[cid])
        overall_baseline.extend(per_class_baseline[cid])

    # -------------------------
    # 5) Prepare results dict
    # -------------------------
    results = {
        "baseline": {"per_class": {}, "overall": {}},
        "default_score": {"per_class": {}, "overall": {}},
        "optimized_score": {"per_class": {}, "overall": {}},
        "optimized_params": {"per_class": {}, "overall": {}},
    }

    # baseline stats
    for cid in range(1, 21):
        results["baseline"]["per_class"][VOC_CLASSES[cid]] = summarize(per_class_baseline[cid])
    results["baseline"]["overall"] = summarize(overall_baseline)

    # -------------------------
    # 6) Default score selection stats
    # -------------------------
    default_w = np.array(args.default_w, dtype=float)
    default_q = float(args.default_qborder)
    default_t = float(args.default_tarea)

    per_class_default: Dict[int, List[float]] = {cid: [] for cid in range(1, 21)}
    overall_default: List[float] = []

    for cid in range(1, 21):
        for cache in per_class_caches[cid]:
            v = select_iou_by_score(cache, default_w, default_q, default_t)
            per_class_default[cid].append(v)
            overall_default.append(v)

    for cid in range(1, 21):
        results["default_score"]["per_class"][VOC_CLASSES[cid]] = summarize(per_class_default[cid])
    results["default_score"]["overall"] = summarize(overall_default)

    # -------------------------
    # 7) Optimization per class + overall
    # -------------------------
    q_range = (args.q_low, args.q_high)
    t_range = (args.t_low, args.t_high)

    per_class_opt_ious: Dict[int, List[float]] = {cid: [] for cid in range(1, 21)}
    overall_opt_ious: List[float] = []

    for cid in range(1, 21):
        cname = VOC_CLASSES[cid]
        caches = per_class_caches[cid]

        if len(caches) < args.min_samples_to_opt:
            results["optimized_params"]["per_class"][cname] = {"note": "too_few_samples", "n": len(caches)}
            results["optimized_score"]["per_class"][cname] = summarize([])
            continue

        train, val = split_train_val(caches, args.opt_split, args.seed + cid * 101)

        w_opt, q_opt, t_opt, train_iou, val_iou = optimize_random(
            caches_train=train,
            caches_val=val,
            seed=args.seed + cid * 101,
            n_random=args.n_random,
            q_range=q_range,
            t_range=t_range,
        )

        ious = [select_iou_by_score(c, w_opt, q_opt, t_opt) for c in caches]
        per_class_opt_ious[cid] = ious
        overall_opt_ious.extend(ious)

        results["optimized_params"]["per_class"][cname] = {
            "w": [float(x) for x in w_opt],
            "q_border": float(q_opt),
            "t_area": float(t_opt),
            "train_meanIoU": float(train_iou),
            "val_meanIoU": float(val_iou),
            "n": len(caches),
        }
        results["optimized_score"]["per_class"][cname] = summarize(ious)

    # overall optimization
    if len(overall_caches) >= args.min_samples_to_opt:
        train, val = split_train_val(overall_caches, args.opt_split, args.seed + 9999)

        w_opt, q_opt, t_opt, train_iou, val_iou = optimize_random(
            caches_train=train,
            caches_val=val,
            seed=args.seed + 9999,
            n_random=args.n_random,
            q_range=q_range,
            t_range=t_range,
        )

        # Evaluate on all overall caches using the overall-opt params
        overall_opt_ious = [select_iou_by_score(c, w_opt, q_opt, t_opt) for c in overall_caches]

        results["optimized_params"]["overall"] = {
            "w": [float(x) for x in w_opt],
            "q_border": float(q_opt),
            "t_area": float(t_opt),
            "train_meanIoU": float(train_iou),
            "val_meanIoU": float(val_iou),
            "n": len(overall_caches),
        }
        results["optimized_score"]["overall"] = summarize(overall_opt_ious)
    else:
        results["optimized_params"]["overall"] = {"note": "too_few_samples", "n": len(overall_caches)}
        results["optimized_score"]["overall"] = summarize([])

    # -------------------------
    # 8) Save results
    # -------------------------
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_path = run_dir / "results.csv"
    headers = ["scope", "setting", "n", "mean", "std", "min", "max"]
    rows = []

    def add_block(setting: str, block: Dict):
        for cid in range(1, 21):
            cname = VOC_CLASSES[cid]
            s = block["per_class"].get(cname, {"n": 0, "mean": 0, "std": 0, "min": 0, "max": 0})
            rows.append([cname, setting, s["n"], s["mean"], s["std"], s["min"], s["max"]])
        so = block.get("overall", {"n": 0, "mean": 0, "std": 0, "min": 0, "max": 0})
        rows.append(["overall", setting, so["n"], so["mean"], so["std"], so["min"], so["max"]])

    add_block("baseline", results["baseline"])
    add_block("default_score", results["default_score"])
    add_block("optimized_score", results["optimized_score"])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

    print(f"\nDone. Outputs saved to: {run_dir}")



def parse_args():
    ap = argparse.ArgumentParser()

    # VOC default path (as you requested)
    ap.add_argument(
        "--voc_root",
        type=str,
        default=r"D:\uwb thesis\RelatedData\archive\VOC2012_train_val\VOC2012_train_val",
        help="VOC root passed to torchvision VOCSegmentation"
    )
    ap.add_argument("--voc_year", type=str, default="2012", choices=["2007", "2012"])
    ap.add_argument("--voc_split", type=str, default="val", choices=["train", "val", "trainval"])
    ap.add_argument("--voc_download", action="store_true")

    # Sampling
    ap.add_argument("--sample_ratio", type=float, default=0.10)
    ap.add_argument("--max_per_class", type=int, default=0, help="0 = no cap")
    ap.add_argument("--seed", type=int, default=0)

    # Models (paths align with your snippet defaults)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    ap.add_argument("--sam2_checkpoint", type=str, default="./checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--sam2_model_config", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")

    ap.add_argument("--gdino_config", type=str, default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    ap.add_argument("--gdino_checkpoint", type=str, default="gdino_checkpoints/groundingdino_swint_ogc.pth")

    ap.add_argument("--dino_box_thresh", type=float, default=0.35)
    ap.add_argument("--dino_text_thresh", type=float, default=0.25)

    ap.add_argument("--sam_multimask", action="store_true", help="SAM2 multimask_output=True (recommended)")

    # Default score params (your requirement)
    ap.add_argument("--default_w", type=float, nargs=4, default=[0.2, 0.2, 0.4, 0.4])
    ap.add_argument("--default_qborder", type=float, default=0.2)
    ap.add_argument("--default_tarea", type=float, default=0.4)

    # Optimization
    ap.add_argument("--n_random", type=int, default=4000)  # VOC太慢，先别上15000
    ap.add_argument("--opt_split", type=float, default=0.8)
    ap.add_argument("--min_samples_to_opt", type=int, default=20)
    ap.add_argument("--q_low", type=float, default=0.05)
    ap.add_argument("--q_high", type=float, default=0.95)
    ap.add_argument("--t_low", type=float, default=0.05)
    ap.add_argument("--t_high", type=float, default=0.95)

    # Output & cache
    ap.add_argument("--out_dir", type=str, default="./voc_runs")
    ap.add_argument("--use_disk_cache", action="store_true", help="Cache per (class, image) candidates to disk .npz")
    ap.add_argument("--dist_dtype", type=str, default="float16", choices=["float16", "float32"])

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
