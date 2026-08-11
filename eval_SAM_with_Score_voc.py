import os
import csv
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from PIL import Image

import cv2

# SAM2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# GroundingDINO 官方 util
from groundingdino.util.inference import load_model as gdino_load_model
from groundingdino.util.inference import load_image as gdino_load_image
from groundingdino.util.inference import predict as gdino_predict

# ============================================================
#  你自己的打分函数：这里请直接替换成【你现在版本】的实现
# ============================================================

# from your_module import compute_scores_new

MIN_AREA_RATIO = 0.1
k_area = 30
t_area = 0.3
W_AREA = 0.2
W_CENTER = 0.2
W_BORDER = 0.5
W_SIL = 0.5
sigmoid_isOn = True


def compute_scores_new(masks_bool, W, H, q_border=0.15, sigmoid_isOn=sigmoid_isOn):
    """
    返回:
        scores: List[float]
        formulas: List[str]  对应每个 mask 的 score 计算公式
    """
    img_area = W * H
    cx, cy = W / 2, H / 2
    dist_map = compute_distance_transform(H, W)
    d_max = dist_map.max() if dist_map.max() > 0 else 1.0

    scores = []
    formulas = []

    for seg in masks_bool:
        area = int(seg.sum())
        if area < MIN_AREA_RATIO * img_area:
            scores.append(0.0)
            formulas.append("score = 0.00 （area too small）")
            continue

        # 面积项
        A_raw = area / img_area
        A_sig = 1.0 / (1.0 + np.exp(-k_area * (A_raw - t_area)))
        # ⚠️ 保留你原来的逻辑
        if not sigmoid_isOn:
            A_sig = A_raw

        # 中心项
        ys, xs = np.where(seg)
        mx, my = xs.mean(), ys.mean()
        Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
        C = 1.0 - np.clip(Dp, 0, 1)

        # 边界项
        q = float(np.quantile(dist_map[seg], q_border))
        Bp = 1.0 - (q / d_max)
        E = 1.0 - np.clip(Bp, 0, 1)

        # 形状项
        S_sil = compute_silhouette_score(seg)

        # 组合得分
        S = W_AREA * A_sig + W_CENTER * C + W_BORDER * E + W_SIL * S_sil

        # 生成公式字符串（带上每一项）
        formula = (
            f"score = {W_AREA:.2f}×{A_sig:.2f} (area) + "
            f"{W_CENTER:.2f}×{C:.2f} (center) + "
            f"{W_BORDER:.2f}×{E:.2f} (border) + "
            f"{W_SIL:.2f}×{S_sil:.2f} (shape) = {S:.3f}"
        )

        scores.append(float(S))
        formulas.append(formula)

    return scores, formulas


# ============================================================
#  VOC2012: 类别 & 数据集工具
# ============================================================

# VOC 原始类名（和标注里的 id 对应）
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

# 给 GroundingDINO 用的更自然的 text prompt
VOC_PROMPTS = {
    1: "airplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "dining table",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "potted plant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tv monitor",
}


def load_voc_image_ids(voc_root: str, split: str = "val") -> List[str]:
    split_file = Path(voc_root) / "ImageSets" / "Segmentation" / f"{split}.txt"
    with open(split_file, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


def load_voc_pair(voc_root: str, image_id: str):
    img_path = Path(voc_root) / "JPEGImages" / f"{image_id}.jpg"
    mask_path = Path(voc_root) / "SegmentationClass" / f"{image_id}.png"
    image = Image.open(img_path).convert("RGB")
    mask_sem = np.array(Image.open(mask_path), dtype=np.uint8)
    return image, mask_sem, str(img_path)


def get_class_specific_mask(mask_sem: np.ndarray, class_ids: List[int]) -> np.ndarray:
    """把若干类别合并为一个前景 mask"""
    mask = np.zeros_like(mask_sem, dtype=bool)
    for cid in class_ids:
        mask |= (mask_sem == cid)
    # 忽略 void=255
    mask &= (mask_sem != 255)
    return mask


# ============================================================
#  工具函数：IoU 计算 & box 转换
# ============================================================

def compute_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    inter = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)

def compute_distance_transform(h, w):
    border_mask = np.zeros((h, w), np.uint8)
    border_mask[1:-1, 1:-1] = 1
    return cv2.distanceTransform(border_mask, distanceType=cv2.DIST_L2, maskSize=5)


def compute_silhouette_score(mask: np.ndarray) -> float:
    mask_u8 = mask.astype(np.uint8)

    # 紧致度
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    P = cv2.arcLength(largest, True)
    A = cv2.contourArea(largest)
    compactness = np.clip((4 * np.pi * A) / (P * P), 0.0, 1.0) if (A > 0 and P > 0) else 0.0

    # 连通分量破碎度
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels > 1:
        largest_area = np.max(stats[1:, cv2.CC_STAT_AREA])
        fragmentation = 1 - (largest_area / np.sum(stats[1:, cv2.CC_STAT_AREA]))
    else:
        fragmentation = 0.0

    # 闭运算检测缝隙
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if closed.sum() == 0:
        gap_ratio = 0.0
    else:
        gap_ratio = 1 - (mask_u8.sum() / closed.sum())

    # 综合
    sil = compactness * (1 - fragmentation) * (1 - gap_ratio)
    return float(np.clip(sil, 0.0, 1.0))
def boxes_cxcywh_to_xyxy_pixels(boxes_cxcywh: torch.Tensor, img_w: int, img_h: int) -> np.ndarray:
    """
    GroundingDINO predict() 返回的是 [cx, cy, w, h]，归一化到 0~1。
    这里转成像素坐标的 xyxy。
    """
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=boxes_cxcywh.dtype)
    boxes = boxes_cxcywh * scale  # 变成像素
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    return xyxy.cpu().numpy()


# ============================================================
#  模型构建：SAM2 + GroundingDINO
# ============================================================

def build_sam2_predictor(model_cfg: str, checkpoint: str, device: str = "cuda") -> SAM2ImagePredictor:
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def build_groundingdino_model(config_path: str, ckpt_path: str, device: str = "cuda"):
    model = gdino_load_model(config_path, ckpt_path, device=device)
    model.eval()
    return model


# ============================================================
#  主评估逻辑：Grounded SAM2 + VOC + 你的 score
# ============================================================

def evaluate_grounded_sam2_on_voc(
    voc_root: str,
    sam2_cfg: str,
    sam2_ckpt: str,
    gdino_cfg: str,
    gdino_ckpt: str,
    class_ids: List[int],
    split: str = "val",
    device: str = "cuda",
    max_images: int = None,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    output_csv: str | None = "groundedsam2_voc_results.csv",
    seed: int = 0,
    min_gt_area_ratio: float | None = None,
    write_csv: bool = True,           # ⭐ 新增
):
    """
    对 VOC + GroundingDINO + SAM2 做评估：

    - class_ids: [7] 表示 car；也可多类 [7,15]。
    - min_gt_area_ratio:
        * None: 不筛选，所有含该类的图片都参与（你的“默认基线实验”）
        * 比如 0.15: 只保留 GT mask 占整图 >= 15% 的图片（大物体实验）
    """

    # 设备
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # 构建模型
    print("Loading GroundingDINO...")
    gdino_model = build_groundingdino_model(gdino_cfg, gdino_ckpt, device=device)

    print("Loading SAM2...")
    sam2_predictor = build_sam2_predictor(sam2_cfg, sam2_ckpt, device=device)

    # ==== 全量读 id ====
    all_ids = load_voc_image_ids(voc_root, split)
    print(f"Found {len(all_ids)} images in VOC2012 {split} set.")

    # ==== 先筛一遍：必须包含该类 & 面积条件（如果给了 min_gt_area_ratio） ====
    print("Scanning images to find those that contain the target classes"
          + (f" with area_ratio >= {min_gt_area_ratio}" if min_gt_area_ratio is not None else "")
          + " ...")

    images_with_class_and_area = []
    for img_id in all_ids:
        _, mask_sem, _ = load_voc_pair(voc_root, img_id)
        gt_mask_tmp = get_class_specific_mask(mask_sem, class_ids)
        gt_area_tmp = int(gt_mask_tmp.sum())
        if gt_area_tmp == 0:
            continue

        if min_gt_area_ratio is not None:
            H_tmp, W_tmp = mask_sem.shape
            area_ratio = gt_area_tmp / float(H_tmp * W_tmp)
            if area_ratio < min_gt_area_ratio:
                continue

        images_with_class_and_area.append(img_id)

    print(f"Images containing class_ids {class_ids}"
          + (f" and area_ratio >= {min_gt_area_ratio}" if min_gt_area_ratio is not None else "")
          + f": {len(images_with_class_and_area)}")

    if len(images_with_class_and_area) == 0:
        print("No images satisfy the class + area condition. Check class_ids or min_gt_area_ratio.")
        return

    # 子采样
    if max_images is not None and max_images < len(images_with_class_and_area):
        rng = np.random.default_rng(seed)
        image_ids = list(rng.choice(images_with_class_and_area, size=max_images, replace=False))
        print(f"Subsampled to {len(image_ids)} images from those with target class/area.")
    else:
        image_ids = images_with_class_and_area

    # 准备 prompt
    prompt_pieces = [VOC_PROMPTS[cid] + " ." for cid in class_ids]
    text_prompt = " ".join(prompt_pieces)
    class_ids_str = ",".join(str(c) for c in class_ids)
    class_names_str = ",".join(VOC_CLASSES[c] for c in class_ids)

    print(f"Using text prompt: \"{text_prompt}\" for classes [{class_ids_str}] ({class_names_str})")

    # CSV 字段
    fieldnames = [
        "image_id",
        "class_ids",
        "class_names",
        "prompt",
        "num_candidates",
        "gt_area",
        "gt_area_ratio",
        "gt_score",

        "sam_best_box_idx",
        "sam_best_mask_idx",
        "sam_best_gt_iou",
        "sam_best_our_score",
        "sam_best_sam_conf",
        "sam_best_dino_logit",

        "ours_best_box_idx",
        "ours_best_mask_idx",
        "ours_best_gt_iou",
        "ours_best_our_score",
        "ours_best_sam_conf",
        "ours_best_dino_logit",
    ]

    rows: List[Dict[str, Any]] = []

    sam_best_ious_all = []
    ours_best_ious_all = []

    # 调试统计
    n_with_gt = 0          # 有 GT 目标（且满足面积条件）
    n_with_boxes = 0       # GDINO 检测到框
    n_with_candidates = 0  # 最终生成了候选 mask

    for img_id in image_ids:
        # 1) 读 GT
        pil_img, mask_sem, img_path = load_voc_pair(voc_root, img_id)
        H, W = mask_sem.shape
        img_area = float(H * W)

        gt_mask = get_class_specific_mask(mask_sem, class_ids)
        gt_area = int(gt_mask.sum())
        if gt_area == 0:
            continue

        gt_area_ratio = gt_area / img_area
        if min_gt_area_ratio is not None and gt_area_ratio < min_gt_area_ratio:
            continue

        n_with_gt += 1

        gt_scores, _ = compute_scores_new([gt_mask], W=W, H=H)
        gt_score = float(gt_scores[0])

        # 2) GroundingDINO 读图
        image_source, image_dino = gdino_load_image(img_path)
        h_src, w_src = image_source.shape[:2]

        # 尺寸不一时，对齐 GT
        if (h_src, w_src) != (H, W):
            gt_mask = cv2.resize(gt_mask.astype(np.uint8), (w_src, h_src),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
            H, W = h_src, w_src
            img_area = float(H * W)

        # 3) GDINO 检测
        boxes_cxcywh, logits, phrases = gdino_predict(
            model=gdino_model,
            image=image_dino,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )

        if boxes_cxcywh.numel() == 0:
            # 没有检测到
            continue
        n_with_boxes += 1

        boxes_xyxy = boxes_cxcywh_to_xyxy_pixels(boxes_cxcywh, img_w=W, img_h=H)
        dino_logits = logits.cpu().numpy()

        # 4) SAM2 set_image
        image_rgb = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
        sam2_predictor.set_image(image_rgb)

        candidates = []

        for box_idx, (box_xyxy, dino_logit, phrase) in enumerate(
            zip(boxes_xyxy, dino_logits, phrases)
        ):
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
                iou_gt = compute_iou(pred_mask, gt_mask)

                our_scores, _ = compute_scores_new([pred_mask], W=W, H=H)
                our_score = float(our_scores[0])

                candidates.append(
                    dict(
                        image_id=img_id,
                        box_idx=box_idx,
                        mask_idx=m_idx,
                        phrase=phrase,
                        dino_logit=float(dino_logit),
                        sam_conf=float(sam_confs[m_idx]),
                        our_score=our_score,
                        gt_iou=iou_gt,
                    )
                )

        if not candidates:
            continue
        n_with_candidates += 1

        num_candidates = len(candidates)

        # 5) SAM 置信度最优
        sam_best = max(candidates, key=lambda c: c["sam_conf"])
        sam_best_ious_all.append(sam_best["gt_iou"])

        # 6) 你的 score 最优
        ours_best = max(candidates, key=lambda c: c["our_score"])
        ours_best_ious_all.append(ours_best["gt_iou"])

        row = {
            "image_id": img_id,
            "class_ids": class_ids_str,
            "class_names": class_names_str,
            "prompt": text_prompt,
            "num_candidates": num_candidates,
            "gt_area": gt_area,
            "gt_area_ratio": gt_area_ratio,
            "gt_score": gt_score,

            "sam_best_box_idx": sam_best["box_idx"],
            "sam_best_mask_idx": sam_best["mask_idx"],
            "sam_best_gt_iou": sam_best["gt_iou"],
            "sam_best_our_score": sam_best["our_score"],
            "sam_best_sam_conf": sam_best["sam_conf"],
            "sam_best_dino_logit": sam_best["dino_logit"],

            "ours_best_box_idx": ours_best["box_idx"],
            "ours_best_mask_idx": ours_best["mask_idx"],
            "ours_best_gt_iou": ours_best["gt_iou"],
            "ours_best_our_score": ours_best["our_score"],
            "ours_best_sam_conf": ours_best["sam_conf"],
            "ours_best_dino_logit": ours_best["dino_logit"],
        }
        rows.append(row)

    # ==== 写 CSV（可选） ====
    if write_csv and output_csv is not None:
        if os.path.dirname(output_csv):
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    # 统计汇总
    sam_mean_iou = float(np.mean(sam_best_ious_all)) if sam_best_ious_all else 0.0
    ours_mean_iou = float(np.mean(ours_best_ious_all)) if ours_best_ious_all else 0.0

    print("======================================")
    print(f"Images with GT (class {class_ids} & area condition): {n_with_gt}")
    print(f"Images where GDINO found boxes:                   {n_with_boxes}")
    print(f"Images with candidate masks from SAM2:            {n_with_candidates}")
    print(f"Total rows written to CSV:                        {len(rows)}")
    print("--------------------------------------")
    print(f"SAM-best mean IoU   : {sam_mean_iou:.4f}")
    print(f"Yours-best mean IoU : {ours_mean_iou:.4f}")
    print(f"CSV saved to        : {output_csv}")
    print("======================================")

    return {
        "rows": rows,
        "sam_best_mean_iou": sam_mean_iou,
        "ours_best_mean_iou": ours_mean_iou,
    }


# ============================================================
#  示例调用
# ============================================================

if __name__ == "__main__":
    VOC_ROOT = r"D:\uwb thesis\RelatedData\archive\VOC2012_train_val\VOC2012_train_val"  # 改成你的 VOC 根目录
    # SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
    # SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    # GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    # GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
    SAM2_CFG = r"configs/sam2.1/sam2.1_hiera_l.yaml"
    SAM2_CKPT = r"./checkpoints/sam2.1_hiera_large.pt"

    GDINO_CFG = r"grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GDINO_CKPT = r"gdino_checkpoints/groundingdino_swint_ogc.pth"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # print("\n=== Experiment 1: baseline (no area filter, car) ===")
    # _res1 = evaluate_grounded_sam2_on_voc(
    #     voc_root=VOC_ROOT,
    #     sam2_cfg=SAM2_CFG,
    #     sam2_ckpt=SAM2_CKPT,
    #     gdino_cfg=GDINO_CFG,
    #     gdino_ckpt=GDINO_CKPT,
    #     class_ids=[7],                      # 7 = car
    #     split="val",
    #     device=DEVICE,
    #     max_images=30,                      # 随机抽 30 张
    #     box_threshold=0.25,
    #     text_threshold=0.20,
    #     output_csv="exp1_car_baseline.csv",
    #     min_gt_area_ratio=None,            # ⭐ 不筛面积
    # )

    # 2. 大物体实验：car，GT 占比 >= 0.15，再随机抽 30 张
    print("\n=== Experiment 2: large objects only (area_ratio >= 0.15, car) ===")
    _res2 = evaluate_grounded_sam2_on_voc(
        voc_root=VOC_ROOT,
        sam2_cfg=SAM2_CFG,
        sam2_ckpt=SAM2_CKPT,
        gdino_cfg=GDINO_CFG,
        gdino_ckpt=GDINO_CKPT,
        class_ids=[6],                      # cat
        split="val",
        device=DEVICE,
        max_images=30,                      # 在满足 area>=0.15 的图中最多抽 30 张
        box_threshold=0.25,
        text_threshold=0.20,
        output_csv="exp2_bus_large_objects.csv",
        min_gt_area_ratio=0.15,            # ⭐ 只保留 GT 占比 >= 15% 的图片
    )

    # 3. 不同类别 + 大物体：每类抽 5 张，合并写到一个 CSV
    # print("\n=== Experiment 3: per class, large objects (area_ratio >= 0.15, 5 images each, merged CSV) ===")
    # target_class_ids = [3, 7, 8, 12, 15]   # 例如 bird, car, cat, dog, person
    #
    # all_rows: list[dict] = []
    #
    # for cid in target_class_ids:
    #     print(f"\n--- Class {cid} ({VOC_CLASSES[cid]}) ---")
    #     res = evaluate_grounded_sam2_on_voc(
    #         voc_root=VOC_ROOT,
    #         sam2_cfg=SAM2_CFG,
    #         sam2_ckpt=SAM2_CKPT,
    #         gdino_cfg=GDINO_CFG,
    #         gdino_ckpt=GDINO_CKPT,
    #         class_ids=[cid],
    #         split="val",
    #         device=DEVICE,
    #         max_images=5,                  # 每类抽 5 张（满足 area>=0.15）
    #         box_threshold=0.25,
    #         text_threshold=0.20,
    #         output_csv=None,               # ⭐ 不在这里单独写文件
    #         min_gt_area_ratio=0.15,
    #         write_csv=False,               # ⭐ 只返回 rows，不写 CSV
    #     )
    #
    #     # 把这一类的 rows 全部加入总列表
    #     all_rows.extend(res["rows"])
    #
    # # 统一写一个 CSV
    # merged_csv = "exp3_all_classes_large5.csv"
    # if all_rows:
    #     fieldnames = list(all_rows[0].keys())
    #
    #     with open(merged_csv, "w", newline="", encoding="utf-8") as f:
    #         writer = csv.DictWriter(f, fieldnames=fieldnames)
    #         writer.writeheader()
    #         for r in all_rows:
    #             writer.writerow(r)
    #
    #     print(f"\nMerged CSV for Experiment 3 saved to: {merged_csv}")
    # else:
    #     print("\nExperiment 3: no rows collected (check class_ids / min_gt_area_ratio).")
