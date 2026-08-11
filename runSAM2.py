import os
import cv2
import torch
import numpy as np
from pathlib import Path

from torchvision.ops import box_convert
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# =============================
# 🔧 User Config
# =============================
IMG_DIR = r"D:\uwb thesis\RelatedData\M60\M60"          # 🔴 输入图片文件夹
OUT_DIR = r"D:\uwb thesis\RelatedData\M60\M60\Masks"          # 🔴 输出 mask 文件夹
TEXT_PROMPT = "Tank ."                      # 🔴 文字 prompt（GroundingDINO）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SAM2
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Grounding DINO
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# =============================
# 🚀 Load Models
# =============================
print("Loading SAM2...")
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

print("Loading GroundingDINO...")
gdino = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE,
)

torch.set_float32_matmul_precision("high")

# =============================
# 🖼 Process One Image
# =============================
def process_image(img_path: Path):
    print(f"Processing: {img_path.name}")

    image_source, image = load_image(str(img_path))
    sam2_predictor.set_image(image_source)

    boxes, _, _ = predict(
        model=gdino,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE,
    )

    if boxes.shape[0] == 0:
        print("  ⚠️ No box detected, skip.")
        return

    h, w, _ = image_source.shape
    boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
    boxes_xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

    # ---- SAM2 segmentation ----
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        masks, scores, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_xyxy,
            multimask_output=True,   # SAM2 默认 3 个
        )

    # ---- 取 SAM2 默认 Top-1 ----
    # masks: [B,3,H,W] → flatten
    masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
    scores = scores.reshape(-1)

    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx].astype(np.uint8) * 255

    out_path = Path(OUT_DIR) / f"{img_path.stem}.jpg.png"
    cv2.imwrite(str(out_path), best_mask)
    print(f"  ✅ Saved: {out_path.name}")

# =============================
# 📁 Batch Loop
# =============================
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp"}
images = [p for p in Path(IMG_DIR).iterdir() if p.suffix.lower() in SUPPORTED]

print(f"Found {len(images)} images.")
for img in images:
    process_image(img)

print("🎉 Done.")
