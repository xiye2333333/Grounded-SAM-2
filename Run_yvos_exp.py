import os, json, math, random, argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

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
    m = np.asfortranarray(mask_bool.astype(np.uint8))
    rle = mask_util.encode(m)
    rle["counts"] = rle["counts"].decode("utf-8")
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": rle["counts"]}

def rle_to_mask(rle_obj: Dict[str, Any]) -> np.ndarray:
    rle = {"size": rle_obj["size"], "counts": rle_obj["counts"].encode("utf-8")}
    m = mask_util.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)

def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


# =========================================================
# YouTube-VOS paths + meta
# Root example:
#   D:\uwb thesis\RelatedData\train\train
#     JPEGImages/<video>/<frame>.jpg
#     Annotations/<video>/<frame>.png
#     meta.json
# =========================================================
def find_yvos_root(root: str) -> str:
    rootp = Path(root)
    if (rootp / "JPEGImages").is_dir() and (rootp / "Annotations").is_dir() and (rootp / "meta.json").is_file():
        return str(rootp)
    # allow one nesting
    rootp2 = rootp / "train"
    if (rootp2 / "JPEGImages").is_dir() and (rootp2 / "Annotations").is_dir() and (rootp2 / "meta.json").is_file():
        return str(rootp2)
    raise FileNotFoundError(f"Could not locate YT-VOS root with JPEGImages/ Annotations/ meta.json under: {root}")

def load_meta(yvos_root: str) -> Dict[str, Any]:
    return json.loads((Path(yvos_root) / "meta.json").read_text(encoding="utf-8"))

def list_single_object_videos(meta: Dict[str, Any]) -> List[Tuple[str, str, str, List[str]]]:
    """
    Returns list of (video_id, obj_id, category, frames_list)
    Only keep videos with exactly 1 object in meta.
    """
    out = []
    videos = (meta.get("videos") or {})
    for vid, vinfo in videos.items():
        objs = (vinfo.get("objects") or {})
        if len(objs) != 1:
            continue
        obj_id = next(iter(objs.keys()))
        oinfo = objs[obj_id]
        category = str(oinfo.get("category", "")).strip()
        frames = oinfo.get("frames", [])
        if not category or not frames:
            continue
        frames = sorted([str(f) for f in frames])  # frames are like "00000"
        out.append((vid, obj_id, category, frames))
    out.sort(key=lambda x: x[0])
    return out

def yvos_img_path(yvos_root: str, video_id: str, frame_id: str) -> str:
    # Usually jpg. Some releases use png; we try both.
    pjpg = Path(yvos_root) / "JPEGImages" / video_id / f"{frame_id}.jpg"
    if pjpg.exists():
        return str(pjpg)
    pjpeg = Path(yvos_root) / "JPEGImages" / video_id / f"{frame_id}.jpeg"
    if pjpeg.exists():
        return str(pjpeg)
    ppng = Path(yvos_root) / "JPEGImages" / video_id / f"{frame_id}.png"
    return str(ppng)

def yvos_anno_path(yvos_root: str, video_id: str, frame_id: str) -> str:
    # Usually png
    p = Path(yvos_root) / "Annotations" / video_id / f"{frame_id}.png"
    if p.exists():
        return str(p)
    # fallback (rare)
    p2 = Path(yvos_root) / "Annotations" / video_id / f"{frame_id}.jpg"
    return str(p2)

def load_yvos_gt_bool(anno_path: str, obj_id: str) -> np.ndarray:
    """
    YouTube-VOS annotation is typically indexed mask:
      background=0, object id = 1/2/3...
    We keep ONLY the single object id.
    """
    arr = np.array(Image.open(anno_path))
    if arr.ndim == 3:
        # sometimes palette can become RGB depending on loader; convert by taking first channel
        arr = arr[:, :, 0]
    oid = int(obj_id)
    return (arr.astype(np.int32) == oid)


def prompt_from_category(category: str) -> str:
    # DINO likes "xxx ."
    return f"{category.strip()} ."


# =========================================================
# Feature functions (same as yours)
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

    A_raw = area_px / img_area
    A = area_term_parabola(A_raw, t_area)

    ys, xs = np.where(mask_bool)
    mx, my = xs.mean(), ys.mean()
    Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
    C = 1.0 - float(np.clip(Dp, 0.0, 1.0))

    q = float(np.quantile(dist_map[mask_bool], q_border))
    E = float(np.clip(q / d_max, 0.0, 1.0))

    Sil = compute_silhouette_score_v2(mask_bool)
    return np.array([A, C, E, Sil], dtype=float)


# =========================================================
# SAM2 + DINO inference for one image (multimask, all masks saved)
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

    B, M, _, _ = masks.shape

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
# Output dirs + saving
# =========================================================
def ensure_dirs(seq_out_dir: Path):
    (seq_out_dir / "json").mkdir(parents=True, exist_ok=True)
    (seq_out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (seq_out_dir / "gt_json").mkdir(parents=True, exist_ok=True)
    (seq_out_dir / "selected_by_score").mkdir(parents=True, exist_ok=True)
    (seq_out_dir / "selected_by_sam").mkdir(parents=True, exist_ok=True)  # baseline png

def result_json_path(seq_out_dir: Path, frame_id: str) -> Path:
    return seq_out_dir / "json" / f"{frame_id}_result.json"

def gt_json_path(seq_out_dir: Path, frame_id: str) -> Path:
    return seq_out_dir / "gt_json" / f"{frame_id}_gt.json"

def save_all_masks_png(seq_out_dir: Path, frame_id: str, annotations: List[Dict[str, Any]]):
    for ann in annotations:
        mid = ann["id"]
        m = rle_to_mask(ann["segmentation"])
        outp = seq_out_dir / "masks" / f"{frame_id}_m{mid:04d}.png"
        if outp.exists():
            continue
        cv2.imwrite(str(outp), (m.astype(np.uint8) * 255))

def build_and_save_gt_json(seq_out_dir: Path, anno_path: str, img_path: str, frame_id: str, obj_id: str, category: str):
    outp = gt_json_path(seq_out_dir, frame_id)
    if outp.exists():
        return
    gt_bool = load_yvos_gt_bool(anno_path, obj_id=obj_id)
    rle = mask_to_rle(gt_bool)
    h, w = gt_bool.shape[:2]
    obj = {
        "image_path": img_path,
        "gt_mask_path": anno_path,
        "width": int(w),
        "height": int(h),
        "video_id": seq_out_dir.name,
        "object_id": str(obj_id),
        "category": category,
        "segmentation_rle": rle,
    }
    outp.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# =========================================================
# Dataset load (reads saved json + gt_json)
# =========================================================
def load_gt_mask_from_gt_json(gt_json_p: Path) -> Optional[np.ndarray]:
    with open(gt_json_p, "r", encoding="utf-8") as f:
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
    gt_index = {p.stem.replace("_gt", ""): p for p in gt_dir.glob("*_gt.json")}

    for fp in json_files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue

        anns = data.get("annotations", [])
        if not anns:
            continue

        key = fp.stem.replace("_result", "")  # frame_id
        gt_json = gt_index.get(key)
        if gt_json is None or (not gt_json.exists()):
            continue

        gt_mask = load_gt_mask_from_gt_json(gt_json)
        if gt_mask is None:
            continue

        H = int(data.get("img_height", gt_mask.shape[0]))
        W = int(data.get("img_width", gt_mask.shape[1]))
        if gt_mask.shape[0] != H or gt_mask.shape[1] != W:
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

    return dataset


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


def optimize_for_video(seq_out_dir: Path, seed: int, gt_ratio: float,
                       n_random: int, q_low: float, q_high: float, t_low: float, t_high: float):
    outp = seq_out_dir / "optimized_params.json"
    if outp.exists():
        return

    dataset = load_seq_dataset(seq_out_dir)
    if len(dataset) < 8:
        return

    rnd = random.Random(seed)
    rnd.shuffle(dataset)
    n_use = max(8, int(round(len(dataset) * gt_ratio)))
    subset = dataset[:n_use]

    rnd.shuffle(subset)
    n_train = max(1, int(len(subset) * 0.8))
    train = subset[:n_train]
    val = subset[n_train:] if len(subset) > n_train else train

    q_range = (min(q_low, q_high), max(q_low, q_high))
    t_range = (min(t_low, t_high), max(t_low, t_high))

    w_opt, q_opt, t_opt, train_iou, val_iou = optimize_random(
        train, val,
        seed=seed,
        n_random=n_random,
        q_range=q_range,
        t_range=t_range
    )

    obj = {
        "video_id": seq_out_dir.name,
        "n_frames_with_gt": len(dataset),
        "n_used_for_opt": len(subset),
        "optimized": {
            "w": w_opt.tolist(),
            "q_border": float(q_opt),
            "t_area": float(t_opt),
            "train_meanIoU": float(train_iou),
            "val_meanIoU": float(val_iou),
        }
    }
    outp.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def evaluate_video(seq_out_dir: Path):
    outp = seq_out_dir / "eval_summary.json"
    if outp.exists():
        return

    param_path = seq_out_dir / "optimized_params.json"
    if not param_path.exists():
        return
    params = json.loads(param_path.read_text(encoding="utf-8"))
    w = normalize_weights(np.asarray(params["optimized"]["w"], dtype=float))
    q_border = float(params["optimized"]["q_border"])
    t_area = float(params["optimized"]["t_area"])

    dataset = load_seq_dataset(seq_out_dir)
    if not dataset:
        return

    baseline_ious, scored_ious = [], []
    best_improve = {"key": None, "delta": -1e9}
    worst_drop = {"key": None, "delta": 1e9}
    per_frame = []

    for item in tqdm(dataset, desc=f"Eval {seq_out_dir.name}", leave=False):
        key = item["key"]
        W, H = item["W"], item["H"]
        dist_map, d_max = item["dist_map"], item["d_max"]
        gt = item["gt"]
        masks = item["masks"]
        raw_scores = item["raw_scores"]

        bidx = int(np.argmax(raw_scores))
        biou = compute_iou(masks[bidx], gt)

        # save SAM-best mask
        sam_png = seq_out_dir / "selected_by_sam" / f"{key}_sam_best.png"
        if not sam_png.exists():
            cv2.imwrite(str(sam_png), (masks[bidx].astype(np.uint8) * 255))

        X = np.stack([mask_features(m, W, H, dist_map, d_max, q_border, t_area) for m in masks], axis=0)
        s_scores = X @ w
        sidx = int(np.argmax(s_scores))
        siou = compute_iou(masks[sidx], gt)

        score_png = seq_out_dir / "selected_by_score" / f"{key}_best.png"
        if not score_png.exists():
            cv2.imwrite(str(score_png), (masks[sidx].astype(np.uint8) * 255))

        delta = siou - biou
        if delta > best_improve["delta"]:
            best_improve = {"key": key, "delta": float(delta), "baseline_iou": float(biou), "scored_iou": float(siou)}
        if delta < worst_drop["delta"]:
            worst_drop = {"key": key, "delta": float(delta), "baseline_iou": float(biou), "scored_iou": float(siou)}

        baseline_ious.append(biou)
        scored_ious.append(siou)
        per_frame.append({
            "frame": key,
            "baseline_best_idx": bidx,
            "scored_best_idx": sidx,
            "baseline_iou": float(biou),
            "scored_iou": float(siou),
            "delta": float(delta),
            "sam_best_mask_png": str(sam_png),
            "score_best_mask_png": str(score_png),
        })

    arr_b = np.asarray(baseline_ious, dtype=float)
    arr_s = np.asarray(scored_ious, dtype=float)

    def stats(a):
        return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()), "n": int(a.size)} if a.size else {}

    summary = {
        "video_id": seq_out_dir.name,
        "n_frames_evaluated": len(per_frame),
        "baseline_iou": stats(arr_b),
        "scored_iou": stats(arr_s),
        "best_improvement": best_improve,
        "worst_drop": worst_drop,
        "per_frame": per_frame,
    }
    outp.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def process_video(
    yvos_root: str,
    video_id: str,
    obj_id: str,
    category: str,
    frames: List[str],
    out_root: Path,
    gdino,
    sam2_predictor,
    device: str,
    box_threshold: float,
    text_threshold: float,
):
    seq_out_dir = out_root / video_id
    ensure_dirs(seq_out_dir)

    prompt = prompt_from_category(category)

    for frame_id in tqdm(frames, desc=f"SAM2+DINO {video_id}", leave=False):
        out_json = result_json_path(seq_out_dir, frame_id)
        if out_json.exists():
            # ensure gt exists
            imgp = yvos_img_path(yvos_root, video_id, frame_id)
            annp = yvos_anno_path(yvos_root, video_id, frame_id)
            if Path(annp).exists():
                build_and_save_gt_json(seq_out_dir, annp, imgp, frame_id, obj_id=obj_id, category=category)
            continue

        imgp = yvos_img_path(yvos_root, video_id, frame_id)
        annp = yvos_anno_path(yvos_root, video_id, frame_id)
        if not Path(imgp).exists() or not Path(annp).exists():
            continue

        data = run_sam2_dino_multimask(
            gdino=gdino,
            sam2_predictor=sam2_predictor,
            img_path=imgp,
            text_prompt=prompt,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # store meta info (helpful for debugging)
        data["video_id"] = video_id
        data["object_id"] = obj_id
        data["category"] = category
        data["frame_id"] = frame_id

        out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")

        anns = data.get("annotations", [])
        if anns:
            save_all_masks_png(seq_out_dir, frame_id, anns)

        build_and_save_gt_json(seq_out_dir, annp, imgp, frame_id, obj_id=obj_id, category=category)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yvos_root", type=str, default=r"D:\uwb thesis\RelatedData\train\train")
    ap.add_argument("--out_root", type=str, default=r"D:\uwb thesis\RelatedData\train\train\SAM_results_singleobj")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sam_ckpt", type=str, default="./checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--sam_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--dino_cfg", type=str, default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    ap.add_argument("--dino_ckpt", type=str, default="gdino_checkpoints/groundingdino_swint_ogc.pth")
    ap.add_argument("--box_th", type=float, default=0.20)
    ap.add_argument("--text_th", type=float, default=0.20)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gt_ratio", type=float, default=0.15)
    ap.add_argument("--n_random", type=int, default=10000)
    ap.add_argument("--q_low", type=float, default=0.05)
    ap.add_argument("--q_high", type=float, default=0.95)
    ap.add_argument("--t_low", type=float, default=0.05)
    ap.add_argument("--t_high", type=float, default=0.95)

    ap.add_argument("--only_video", type=str, default="", help="debug: run only one video id")
    args = ap.parse_args()

    yvos_root = find_yvos_root(args.yvos_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    meta = load_meta(yvos_root)
    vids = list_single_object_videos(meta)

    if args.only_video:
        vids = [x for x in vids if x[0] == args.only_video]
        if not vids:
            raise SystemExit(f"Video not found or not single-object: {args.only_video}")

    # load models once
    print("Loading SAM2...")
    sam2_model = build_sam2(args.sam_cfg, args.sam_ckpt, device=args.device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    print("Loading GroundingDINO...")
    gdino = load_model(args.dino_cfg, args.dino_ckpt, device=args.device)

    torch.set_float32_matmul_precision("high")

    print(f"Single-object videos: {len(vids)} (multi-object videos skipped)")

    for (video_id, obj_id, category, frames) in tqdm(vids, desc="YT-VOS videos"):
        process_video(
            yvos_root=yvos_root,
            video_id=video_id,
            obj_id=obj_id,
            category=category,
            frames=frames,
            out_root=out_root,
            gdino=gdino,
            sam2_predictor=sam2_predictor,
            device=args.device,
            box_threshold=args.box_th,
            text_threshold=args.text_th,
        )

        # optimize + eval
        optimize_for_video(
            seq_out_dir=out_root / video_id,
            seed=args.seed,
            gt_ratio=args.gt_ratio,
            n_random=args.n_random,
            q_low=args.q_low, q_high=args.q_high,
            t_low=args.t_low, t_high=args.t_high,
        )
        evaluate_video(out_root / video_id)

    print("[DONE] All single-object videos processed.")


if __name__ == "__main__":
    main()