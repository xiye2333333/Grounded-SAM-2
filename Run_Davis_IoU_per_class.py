import os
import re
import json
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
from torchvision.ops import box_convert
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# =========================
# Utils
# =========================
def sanitize_for_folder(name: str, max_len: int = 80) -> str:
    """
    Turn prompt into a filesystem-safe folder name.
    """
    name = name.strip()
    name = re.sub(r"[<>:\"/\\|?*\n\r\t]+", "_", name)
    name = re.sub(r"\s+", " ", name)
    name = name.replace(" .", ".")
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    if not name:
        name = "prompt"
    return name

def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


# =========================
# DAVIS paths
# =========================
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
    # DAVIS annotation is usually indexed labels, >0 is foreground
    if m.ndim == 2:
        fg = (m > 0)
    else:
        fg = (m.sum(axis=2) > 0)
    return fg.astype(bool)


# =========================
# DINO + SAM2 inference (best-mask only)
# =========================
@torch.no_grad()
def infer_best_mask_for_image(
    gdino,
    sam2_predictor,
    img_path: str,
    text_prompt: str,
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Returns:
      best_mask_bool (HxW) or None
      note string
    Strategy:
      - DINO predicts boxes
      - SAM2 predicts multimasks for each box
      - choose the single mask with max SAM score among all boxes+masks
    """
    image_source, image = load_image(str(img_path))  # image_source: HWC BGR/RGB depending on util; SAM2 predictor expects numpy image
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
        return None, "no_box"

    boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
    boxes_xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

    # SAM2 (multimask)
    with torch.autocast(device_type=device, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32):
        masks, scores, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_xyxy,
            multimask_output=True,
        )

    # normalize shapes:
    # masks: [B,M,H,W] or [M,H,W]
    if masks is None:
        return None, "sam_no_mask"
    if masks.ndim == 3:
        masks = masks[None, ...]
    if scores is None:
        # if wrapper returns None, fallback to first mask
        return masks[0, 0].astype(bool), "sam_scores_none"

    if scores.ndim == 1:
        scores = scores[None, ...]

    B, M, Hm, Wm = masks.shape

    # pick global best by SAM score
    best_s = -1e9
    best_m = None
    for b in range(B):
        for r in range(M):
            s = float(scores[b, r])
            if s > best_s:
                best_s = s
                best_m = masks[b, r].astype(bool)

    if best_m is None:
        return None, "no_best"
    return best_m, f"ok_best_sam_score={best_s:.4f}"


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser("DAVIS DINO+SAM2 best-mask + IoU")

    # required by you
    ap.add_argument("--seq", type=str, default="drift-turn", help="DAVIS sequence name, e.g. bmx-bump")
    ap.add_argument("--prompt", type=str, default="Drifting Car .", help='Text prompt for DINO, e.g. "bmx rider ."')
    ap.add_argument("--conf_th", type=float, default=0.2, help="Confidence threshold (used for both box_th and text_th)")

    # fixed defaults (you can still override if needed)
    ap.add_argument("--davis_root", type=str, default=r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS")
    ap.add_argument("--resolution", type=str, default="480p", choices=["480p", "1080p"])
    ap.add_argument("--out_root", type=str, default=r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results")

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sam_ckpt", type=str, default="./checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--sam_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--dino_cfg", type=str, default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    ap.add_argument("--dino_ckpt", type=str, default="gdino_checkpoints/groundingdino_swint_ogc.pth")

    args = ap.parse_args()

    davis_root = find_davis_root(args.davis_root)

    # validate seq exists
    seqs = list_sequences(davis_root, args.resolution)
    if args.seq not in seqs:
        raise SystemExit(f"Sequence not found in DAVIS/{args.resolution}: {args.seq}\n"
                         f"Example: {seqs[:20]} ... (total {len(seqs)})")

    # output dir: <out_root>\<seq>\SAM_<prompt>
    prompt_dirname = sanitize_for_folder(args.prompt)
    out_dir = Path(args.out_root) / args.seq / f"SAM_{prompt_dirname}"
    pred_dir = out_dir / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # save run config
    (out_dir / "run_args.json").write_text(
        json.dumps(
            {
                "seq": args.seq,
                "prompt": args.prompt,
                "conf_th": args.conf_th,
                "resolution": args.resolution,
                "davis_root": davis_root,
                "out_dir": str(out_dir),
                "device": args.device,
                "sam_ckpt": args.sam_ckpt,
                "sam_cfg": args.sam_cfg,
                "dino_cfg": args.dino_cfg,
                "dino_ckpt": args.dino_ckpt,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # load models
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

    frames = list_frames(davis_root, args.resolution, args.seq)
    if not frames:
        raise SystemExit(f"No frames found for {args.seq} in {args.resolution}")

    # run
    per_frame = []
    for ff in tqdm(frames, desc=f"Run {args.seq}", leave=True):
        imgp = davis_img_path(davis_root, args.resolution, args.seq, ff)
        gtp = davis_gt_path(davis_root, args.resolution, args.seq, ff)

        if gtp is None:
            # DAVIS normally has GT for all frames, but we handle missing gracefully
            per_frame.append({"frame": Path(ff).stem, "iou": None, "note": "no_gt"})
            continue

        gt_bool = load_davis_gt_bool(gtp)

        # predict best mask
        pred_bool, note = infer_best_mask_for_image(
            gdino=gdino,
            sam2_predictor=sam2_predictor,
            img_path=imgp,
            text_prompt=args.prompt,
            device=args.device,
            box_threshold=args.conf_th,
            text_threshold=args.conf_th,
        )

        stem = Path(ff).stem
        pred_png = pred_dir / f"{stem}.png"

        if pred_bool is None:
            # save empty mask for reproducibility
            empty = np.zeros_like(gt_bool, dtype=np.uint8)
            cv2.imwrite(str(pred_png), empty)
            per_frame.append({"frame": stem, "iou": 0.0, "note": note, "pred_png": str(pred_png), "gt_path": gtp})
            continue

        # ensure same shape
        if pred_bool.shape != gt_bool.shape:
            # try to resize with nearest if mismatch (should not happen normally)
            pred_u8 = pred_bool.astype(np.uint8)
            pred_u8 = cv2.resize(pred_u8, (gt_bool.shape[1], gt_bool.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred_bool = pred_u8.astype(bool)

        iou = compute_iou(pred_bool, gt_bool)

        cv2.imwrite(str(pred_png), (pred_bool.astype(np.uint8) * 255))

        per_frame.append(
            {"frame": stem, "iou": float(iou), "note": note, "pred_png": str(pred_png), "gt_path": gtp}
        )

    # stats
    valid_ious = [r["iou"] for r in per_frame if isinstance(r["iou"], (int, float))]
    mean_iou = float(np.mean(valid_ious)) if valid_ious else None
    min_iou = float(np.min(valid_ious)) if valid_ious else None
    max_iou = float(np.max(valid_ious)) if valid_ious else None

    summary = {
        "seq": args.seq,
        "prompt": args.prompt,
        "conf_th": float(args.conf_th),
        "resolution": args.resolution,
        "n_frames": len(per_frame),
        "n_iou_valid": len(valid_ious),
        "mean_iou": mean_iou,
        "min_iou": min_iou,
        "max_iou": max_iou,
        "out_dir": str(out_dir),
        "pred_dir": str(pred_dir),
    }

    # write outputs
    (out_dir / "iou_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # csv (no pandas dependency)
    csv_path = out_dir / "iou_per_frame.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("frame,iou,note,pred_png,gt_path\n")
        for r in per_frame:
            iou_str = "" if r["iou"] is None else f"{r['iou']:.6f}"
            f.write(f"{r['frame']},{iou_str},{r.get('note','')},{r.get('pred_png','')},{r.get('gt_path','')}\n")

    # also dump per-frame json (optional but handy)
    (out_dir / "per_frame.json").write_text(json.dumps(per_frame, indent=2), encoding="utf-8")

    print("[DONE] IoU summary:")
    print(json.dumps(summary, indent=2))
    print(f"[DONE] Saved to: {out_dir}")


if __name__ == "__main__":
    main()