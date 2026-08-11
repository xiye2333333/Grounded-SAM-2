"""
预计算脚本：
- 对指定 VOC 类别，抽样若干张图片（比如 30）
- 使用 GroundingDINO + SAM2 生成 candidate masks
- 将每张图的：
    * gt_mask（用于该类别的语义前景）
    * 所有候选 masks (K, H, W)
    * sam_confs (K,)
    * dino_logits (K,)
    * box_indices (K,)
  保存到 .npz，供之后离线调参使用
"""

import os
import random
from pathlib import Path

import numpy as np
import cv2
import torch

import eval_SAM_with_Score as bench  # 你当前那份大脚本

# ============ 路径配置（按你的环境修改） ============

VOC_ROOT = r"D:\uwb thesis\RelatedData\archive\VOC2012_train_val\VOC2012_train_val"

SAM2_CFG = r"configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = r"./checkpoints/sam2.1_hiera_large.pt"

GDINO_CFG = r"grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_CKPT = r"gdino_checkpoints/groundingdino_swint_ogc.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLIT = "val"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20
MIN_GT_AREA_RATIO = 0.0  # 只保留 GT 占比 >= 15% 的图（大物体）
MAX_IMAGES_PER_CLASS = 30  # 每个类别最多抽 30 张

# 输出目录：每个类别一个子目录
OUT_ROOT = Path("precomputed_masks")  # 之后调参脚本就从这里读 npz


def precompute_for_class(class_id: int, max_images: int = MAX_IMAGES_PER_CLASS, seed: int = 0):
    """
    为一个 VOC 类别预计算：
    - 抽样 max_images 张图（满足含该类 & 面积条件）
    - 对每张图跑 GDINO+SAM2，保存候选 masks 和 GT mask
    """
    random.seed(seed)
    np.random.seed(seed)

    OUT_DIR = OUT_ROOT / f"class_{class_id}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 构建模型
    print(f"\n[Class {class_id}] Loading models ...")
    gdino_model = bench.build_groundingdino_model(GDINO_CFG, GDINO_CKPT, device=DEVICE)
    sam2_predictor = bench.build_sam2_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)

    # 2) 获取所有 image_ids
    all_ids = bench.load_voc_image_ids(VOC_ROOT, SPLIT)
    print(f"[Class {class_id}] Found {len(all_ids)} images in VOC {SPLIT} set.")

    # 3) 先筛一遍：包含该类 & 面积 >= MIN_GT_AREA_RATIO
    #    注意：我们一次只处理一个 class_id，这里 class_ids=[class_id]
    candidate_ids = []
    for img_id in all_ids:
        _, mask_sem, _ = bench.load_voc_pair(VOC_ROOT, img_id)
        gt_mask_tmp = bench.get_class_specific_mask(mask_sem, [class_id])
        gt_area_tmp = int(gt_mask_tmp.sum())
        if gt_area_tmp == 0:
            continue

        H_tmp, W_tmp = mask_sem.shape
        area_ratio = gt_area_tmp / float(H_tmp * W_tmp)
        if area_ratio < MIN_GT_AREA_RATIO:
            continue

        candidate_ids.append(img_id)

    print(f"[Class {class_id}] Images with target class and area_ratio >= {MIN_GT_AREA_RATIO}: {len(candidate_ids)}")

    if not candidate_ids:
        print(f"[Class {class_id}] No valid images, skip.")
        return

    # 4) 随机抽 max_images
    if len(candidate_ids) > max_images:
        rng = np.random.default_rng(seed)
        image_ids = list(rng.choice(candidate_ids, size=max_images, replace=False))
    else:
        image_ids = candidate_ids

    print(f"[Class {class_id}] Will precompute masks for {len(image_ids)} images.")

    # 5) 构建 text prompt
    prompt = bench.VOC_PROMPTS[class_id] + " ."
    print(f"[Class {class_id}] Using text prompt: \"{prompt}\"")

    # 6) 遍历每张图，生成 & 保存 masks
    done = 0
    for img_id in image_ids:
        print(f"[Class {class_id}] Processing image: {img_id}")

        # 如果已经算过 npz 就跳过（避免重复）
        out_path = OUT_DIR / f"{img_id}.npz"
        if out_path.exists():
            print(f"[Class {class_id}] {img_id}.npz already exists, skip.")
            done += 1
            continue

        # 6.1 读 GT
        pil_img, mask_sem, img_path = bench.load_voc_pair(VOC_ROOT, img_id)
        H, W = mask_sem.shape

        gt_mask = bench.get_class_specific_mask(mask_sem, [class_id])
        gt_area = int(gt_mask.sum())
        if gt_area == 0:
            print(f"  - No GT area for class {class_id}, skip.")
            continue

        # 6.2 GroundingDINO load_image
        image_source, image_dino = bench.gdino_load_image(img_path)
        h_src, w_src = image_source.shape[:2]

        # 对齐 GT 尺寸到 GDINO / SAM 使用的图像尺寸
        if (h_src, w_src) != (H, W):
            gt_mask = cv2.resize(gt_mask.astype(np.uint8), (w_src, h_src),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
            H, W = h_src, w_src

        # 6.3 GDINO 检测 boxes
        boxes_cxcywh, logits, phrases = bench.gdino_predict(
            model=gdino_model,
            image=image_dino,
            caption=prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE,
        )

        if boxes_cxcywh.numel() == 0:
            print("  - GroundingDINO found no boxes, skip.")
            continue

        boxes_xyxy = bench.boxes_cxcywh_to_xyxy_pixels(boxes_cxcywh, img_w=W, img_h=H)
        dino_logits = logits.cpu().numpy()

        # 6.4 SAM2 set_image & 生成 masks
        image_rgb = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
        sam2_predictor.set_image(image_rgb)

        all_masks = []
        all_sam_confs = []
        all_dino_logits = []
        all_box_indices = []

        for box_idx, (box_xyxy, dino_logit) in enumerate(zip(boxes_xyxy, dino_logits)):
            masks_np, sam_confs, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_xyxy[None, :],
                multimask_output=True,
                return_logits=False,
            )
            # masks_np: (C, H, W)
            num_masks = masks_np.shape[0]

            for m_idx in range(num_masks):
                pred_mask = masks_np[m_idx].astype(bool)
                # 可选：过滤掉面积太小的 mask
                if pred_mask.sum() == 0:
                    continue

                all_masks.append(pred_mask)
                all_sam_confs.append(float(sam_confs[m_idx]))
                all_dino_logits.append(float(dino_logit))
                all_box_indices.append(int(box_idx))

        if not all_masks:
            print("  - SAM2 produced no valid masks, skip.")
            continue

        masks_arr = np.stack(all_masks, axis=0).astype(np.uint8)  # (K, H, W)，0/1
        sam_confs_arr = np.array(all_sam_confs, dtype=np.float32)
        dino_logits_arr = np.array(all_dino_logits, dtype=np.float32)
        box_indices_arr = np.array(all_box_indices, dtype=np.int32)
        gt_mask_arr = gt_mask.astype(np.uint8)

        # 6.5 保存 npz
        np.savez_compressed(
            out_path,
            image_id=str(img_id),  # ✅ 直接存字符串
            class_id=np.int32(class_id),
            H=np.int32(H),
            W=np.int32(W),
            gt_mask=gt_mask_arr,  # (H, W), uint8 0/1
            masks=masks_arr,  # (K, H, W), uint8 0/1
            sam_confs=sam_confs_arr,  # (K,)
            dino_logits=dino_logits_arr,  # (K,)
            box_indices=box_indices_arr,  # (K,)
        )

        print(f"  - Saved {masks_arr.shape[0]} masks to {out_path}")
        done += 1

    print(f"\n[Class {class_id}] Done. Saved {done} images to {OUT_DIR}")


if __name__ == "__main__":
    # 举例：先给 "car" 类（7）预计算 30 张
    # precompute_for_class(class_id=7, max_images=30, seed=42)

    # 如果想多类：
    for cid in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        precompute_for_class(class_id=cid, max_images=30, seed=42 + cid)
