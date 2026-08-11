import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import zlib
import base64

# =============================
# ⚙️ User Configurations
# =============================
TEXT_PROMPT = "seat ."
IMG_DIR = r"D:\uwb thesis\code\Grounded-SAM-2\assets\images"

SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = Path("outputs/AllMasks_v2_score_record")
OUTPUT_MASK_DIR = Path("outputs/AllMasks_v2_score")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

DUMP_JSON_RESULTS = True
SAVE_ANNOTATED = True
MULTIMASK_OUTPUT = True

# =============================
# ✅ New scoring hyper-params
# =============================
W_AREA   = 0.2
W_CENTER = 0.2
W_BORDER = 0.6
W_SIL    = 0.2          # NEW: silhouette weight (adjust as needed)
Q_BORDER = 0.25
MIN_AREA_RATIO = 0.01

# Area term: parabola peak location d (0~1)
t_area = 0.40

# =============================
# 🔧 Setup
# =============================
torch.set_float32_matmul_precision("high")

print("🧩 Loading SAM2 model...")
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

print("🦖 Loading Grounding DINO model...")
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE,
)

def pack_lowres_logits(logits_2d: np.ndarray) -> dict:
    """
    logits_2d: (256,256) float (usually float32 from SAM2)
    Returns a JSON-serializable dict with compressed payload.
    """
    arr = logits_2d.astype(np.float16)  # save space
    raw = arr.tobytes()
    comp = zlib.compress(raw, level=9)
    b64 = base64.b64encode(comp).decode("utf-8")
    return {"shape": [arr.shape[0], arr.shape[1]], "dtype": "float16", "zlib_b64": b64}

def unpack_lowres_logits(packed: dict) -> np.ndarray:
    comp = base64.b64decode(packed["zlib_b64"].encode("utf-8"))
    raw = zlib.decompress(comp)
    h, w = packed["shape"]
    arr = np.frombuffer(raw, dtype=np.float16).reshape(h, w).astype(np.float32)
    return arr

# =============================
# 🛠 Scoring functions (NEW)
# =============================
EPS = 1e-8

def compute_distance_transform(h, w):
    border_mask = np.zeros((h, w), np.uint8)
    border_mask[1:-1, 1:-1] = 1
    return cv2.distanceTransform(border_mask, distanceType=cv2.DIST_L2, maskSize=5)

def area_term_parabola(x: float, d: float, eps: float = 1e-6) -> float:
    """
    Piecewise parabola peaked at d:
    f(x)=max(0, 1-((x-d)/d)^2), clipped to [0,1]
    """
    d = float(np.clip(d, eps, 1.0))
    x = float(np.clip(x, 0.0, 1.0))
    val = 1.0 - ((x - d) / d) ** 2
    return float(np.clip(val, 0.0, 1.0))

def normalize_weights(w_area, w_center, w_border, w_sil, eps: float = 1e-8):
    s = float(w_area + w_center + w_border + w_sil)
    if s <= eps:
        return 0.25, 0.25, 0.25, 0.25
    return w_area / s, w_center / s, w_border / s, w_sil / s

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

def compute_custom_scores_new(masks_bool, W, H):
    """
    Returns:
        scores_custom: list[float] in [0,1]
        breakdowns: list[dict] each contains A_raw, A, C, E, Sil and normalized weights
    """
    img_area = W * H
    cx, cy = W / 2, H / 2
    dist_map = compute_distance_transform(H, W)
    d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

    # normalized weights -> convex combination -> score in [0,1]
    wA, wC, wB, wS = normalize_weights(W_AREA, W_CENTER, W_BORDER, W_SIL)

    scores = []
    breakdowns = []

    for seg in masks_bool:
        area_px = int(seg.sum())
        if area_px < MIN_AREA_RATIO * img_area:
            scores.append(0.0)
            breakdowns.append({
                "A_raw": area_px / img_area,
                "A": 0.0, "C": 0.0, "E": 0.0, "Sil": 0.0,
                "wA": wA, "wC": wC, "wB": wB, "wS": wS,
                "note": "area too small"
            })
            continue

        # ---- Area term ----
        A_raw = area_px / img_area
        A = area_term_parabola(A_raw, t_area)

        # ---- Center term ----
        ys, xs = np.where(seg)
        if xs.size == 0:
            scores.append(0.0)
            breakdowns.append({
                "A_raw": A_raw, "A": A, "C": 0.0, "E": 0.0, "Sil": 0.0,
                "wA": wA, "wC": wC, "wB": wB, "wS": wS,
                "note": "empty mask"
            })
            continue
        mx, my = xs.mean(), ys.mean()
        Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
        C = 1.0 - float(np.clip(Dp, 0.0, 1.0))

        # ---- Border term ----
        mask_distances = dist_map[seg]
        if mask_distances.size == 0:
            scores.append(0.0)
            breakdowns.append({
                "A_raw": A_raw, "A": A, "C": C, "E": 0.0, "Sil": 0.0,
                "wA": wA, "wC": wC, "wB": wB, "wS": wS,
                "note": "no distances"
            })
            continue

        q_val = float(np.quantile(mask_distances, Q_BORDER))
        E = float(np.clip(q_val / d_max, 0.0, 1.0))

        # ---- Silhouette term (v2) ----
        Sil = compute_silhouette_score_v2(seg)

        # ---- Final score ----
        score = wA * A + wC * C + wB * E + wS * Sil
        score = float(np.clip(score, 0.0, 1.0))

        scores.append(score)
        breakdowns.append({
            "A_raw": float(A_raw), "A": float(A), "C": float(C), "E": float(E), "Sil": float(Sil),
            "wA": float(wA), "wC": float(wC), "wB": float(wB), "wS": float(wS),
            "note": ""
        })

    return scores, breakdowns

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

# =============================
# 🚀 Main processing
# =============================
def process_image(img_path: str):
    print(f"\n🖼 Processing: {img_path}")
    image_source, image = load_image(img_path)
    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE,
    )

    if boxes.size(0) == 0:
        print(f"⚠️ No object detected for {img_path}")
        return

    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # SAM segmentation
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        masks, scores_sam, low_res_masks = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=True,
        )

    # Expand all masks [B,3,H,W] → [3B,H,W]
    masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
    scores_sam = scores_sam.reshape(-1)
    low_res_masks = low_res_masks.reshape(-1, low_res_masks.shape[-2], low_res_masks.shape[-1])
    input_boxes = np.repeat(input_boxes, 3, axis=0)
    confidences = np.repeat(confidences.numpy(), 3, axis=0)
    class_names = [cls for cls in labels for _ in range(3)]
    masks_bool = masks.astype(bool)

    # ✅ New custom scoring (with breakdown)
    scores_custom, breakdowns = compute_custom_scores_new(masks_bool, w, h)
    best_idx = int(np.argmax(scores_custom))
    best_mask = masks_bool[best_idx]
    best_score = scores_custom[best_idx]
    print(f"🔎 Best mask index = {best_idx}, custom score = {best_score:.3f}")

    # Save best mask
    out_mask_path = OUTPUT_MASK_DIR / f"{Path(img_path).name}.png"
    cv2.imwrite(str(out_mask_path), (best_mask.astype("uint8")) * 255)
    print(f"✅ Saved best mask to {out_mask_path}")

    # Visualization + JSON
    if SAVE_ANNOTATED or DUMP_JSON_RESULTS:
        class_ids = np.arange(len(class_names))
        labels_text = [f"{cls} {conf:.2f}" for cls, conf in zip(class_names, confidences)]
        img_bgr = cv2.imread(img_path)
        detections = sv.Detections(xyxy=input_boxes, mask=masks_bool, class_id=class_ids)

        if SAVE_ANNOTATED:
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            mask_annotator = sv.MaskAnnotator()
            annotated = box_annotator.annotate(scene=img_bgr.copy(), detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels_text)
            annotated = mask_annotator.annotate(scene=annotated, detections=detections)
            cv2.imwrite(str(OUTPUT_DIR / f"{Path(img_path).stem}_annotated.jpg"), annotated)

        if DUMP_JSON_RESULTS:
            mask_rles = [single_mask_to_rle(m.astype(np.uint8)) for m in masks_bool]
            annotations = []
            for i, (cls, box, rle, s_sam, s_custom, br, lr) in enumerate(
                    zip(class_names, input_boxes, mask_rles, scores_sam, scores_custom, breakdowns, low_res_masks)
            ):
                annotations.append({
                    "mask_index": i,
                    "class_name": cls,
                    "bbox": box.tolist(),
                    "segmentation": rle,
                    "score_sam": float(s_sam),
                    "score_custom": float(s_custom),

                    # ✅ NEW: 保存 low-res logits（用于二次 mask prompt）
                    "low_res_logits": pack_lowres_logits(lr),

                    "score_terms": {
                        "A_raw": br["A_raw"],
                        "A": br["A"],
                        "C": br["C"],
                        "E": br["E"],
                        "Sil": br["Sil"],
                        "wA": br["wA"],
                        "wC": br["wC"],
                        "wB": br["wB"],
                        "wS": br["wS"],
                        "note": br.get("note", "")
                    },
                    "is_best": (i == best_idx)
                })

            results = {
                "image_path": img_path,
                "best_mask_index": best_idx,
                "best_score_custom": float(best_score),
                "annotations": annotations,
                "box_format": "xyxy",
                "img_width": w,
                "img_height": h,

                # record score config for reproducibility
                "score_config": {
                    "W_AREA": W_AREA, "W_CENTER": W_CENTER, "W_BORDER": W_BORDER, "W_SIL": W_SIL,
                    "Q_BORDER": Q_BORDER, "MIN_AREA_RATIO": MIN_AREA_RATIO,
                    "t_area": t_area,
                    "weight_normalization": "L1"
                }
            }

            with open(OUTPUT_DIR / f"{Path(img_path).stem}_result.json", "w") as f:
                json.dump(results, f, indent=4)

# =============================
# 📁 Batch Processing Loop
# =============================
SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
images = sorted([
    str(p) for p in Path(IMG_DIR).iterdir()
    if p.suffix.lower() in SUPPORTED_EXTS
])
print(f"Found {len(images)} images in {IMG_DIR}")

for img_path in images:
    process_image(img_path)

print("\n🎉 All images processed successfully!")
