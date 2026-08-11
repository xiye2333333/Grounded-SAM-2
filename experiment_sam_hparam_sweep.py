"""
Low-cost Experiment v3 (simple version):

- 固定一套 score 函数（来自 eval_SAM_with_Score）
- 一次只扫 SAM/DINO 的一个超参数（比如 box_threshold），设置 3 个档位
- 对每个档位：
    * 在同一批 VOC 图像 (class_id, image_id) 上跑 GroundingDINO + SAM2
    * 保存所有候选 mask 到 npz
- 然后：
    * 对三套 config 的 npz，比较 “SAM Top1（sam_conf 最大）” 的
        - score (compute_scores_new)
        - IoU (vs GT)
    * 在共同样本交集上打印 mean score / mean IoU
    * 写一个 summary CSV

你只需要修改 CONFIG 里的几行参数，然后直接：
    python experiment_sam_hparam_sweep_simple.py
即可。
"""

from pathlib import Path
import csv

import numpy as np
import torch
import cv2
from PIL import Image

# SAM2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# GroundingDINO
from groundingdino.util.inference import load_model as gdino_load_model
from groundingdino.util.inference import load_image as gdino_load_image
from groundingdino.util.inference import predict as gdino_predict

# 你自己的评分 / IoU 模块
import eval_SAM_with_Score as bench


# =========================
# CONFIG：只改这里就能跑
# =========================

CONFIG = {
    # VOC 数据集根目录
    "VOC_ROOT": r"D:\uwb thesis\RelatedData\archive\VOC2012_train_val\VOC2012_train_val",

    # SAM2 / GroundingDINO 的配置与权重路径
    "SAM2_CFG": r"configs/sam2.1/sam2.1_hiera_l.yaml",
    "SAM2_CKPT": r"./checkpoints/sam2.1_hiera_large.pt",
    "GDINO_CFG": r"grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "GDINO_CKPT": r"gdino_checkpoints/groundingdino_swint_ogc.pth",

    # 要评估的 VOC 类别 id（例如 [7] 只看 car；[7, 8] car+cat）
    "CLASS_IDS": [19],

    # 使用的 split
    "SPLIT": "val",

    # 每个 class 最多抽多少张图
    "MAX_IMAGES_PER_CLASS": 30,

    # GT mask 占图像比例 < 此值的样本会被跳过
    "MIN_GT_AREA_RATIO": 0.15,

    # 随机种子（用于从满足条件的图中随机抽样）
    "SEED": 0,

    # 要扫的 SAM/DINO 超参数：
    # 当前支持: "box_threshold" 或 "text_threshold"
    "HPARAM_NAME": "box_threshold",
    # 三个档位的值
    "HPARAM_VALUES": [0.20, 0.25, 0.30],

    # 输出根目录（会在下面生成 {tag}/class_{cid}/*.npz）
    "OUTPUT_ROOT": r"precomputed_hparam_sweep_boxth_simple",

    # 设备
    "DEVICE": "cuda",   # 或 "cpu"

    # score 的 q_border 参数
    "Q_BORDER": 0.15,
}


# =========================
# VOC helpers
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


def load_voc_image_ids(voc_root: Path, split: str = "val"):
    split_file = voc_root / "ImageSets" / "Segmentation" / f"{split}.txt"
    with open(split_file, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


def load_voc_pair(voc_root: Path, image_id: str):
    img_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
    mask_path = voc_root / "SegmentationClass" / f"{image_id}.png"
    image = Image.open(img_path).convert("RGB")
    mask_sem = np.array(Image.open(mask_path), dtype=np.uint8)
    return image, mask_sem, str(img_path)


def get_class_specific_mask(mask_sem: np.ndarray, class_ids):
    mask = np.zeros_like(mask_sem, dtype=bool)
    for cid in class_ids:
        mask |= (mask_sem == cid)
    mask &= (mask_sem != 255)
    return mask


def boxes_cxcywh_to_xyxy_pixels(boxes_cxcywh: torch.Tensor, img_w: int, img_h: int) -> np.ndarray:
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=boxes_cxcywh.dtype, device=boxes_cxcywh.device)
    boxes = boxes_cxcywh * scale
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    return xyxy.cpu().numpy()


# =========================
# Model builders
# =========================

def build_sam2_predictor(model_cfg: str, checkpoint: str, device: str = "cuda") -> SAM2ImagePredictor:
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def build_groundingdino_model(config_path: str, ckpt_path: str, device: str = "cuda"):
    model = gdino_load_model(config_path, ckpt_path, device=device)
    model.eval()
    return model


# =========================
# Image selection (same set)
# =========================

def select_images_for_classes(
    voc_root: Path,
    class_ids,
    split: str,
    min_gt_area_ratio: float,
    max_images_per_class: int,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    all_ids = load_voc_image_ids(voc_root, split)

    pairs = []
    for cid in class_ids:
        valid_imgs = []
        for img_id in all_ids:
            _, mask_sem, _ = load_voc_pair(voc_root, img_id)
            gt_mask = get_class_specific_mask(mask_sem, [cid])
            gt_area = int(gt_mask.sum())
            if gt_area == 0:
                continue
            H, W = mask_sem.shape
            if gt_area / float(H * W) < min_gt_area_ratio:
                continue
            valid_imgs.append(img_id)

        if not valid_imgs:
            print(f"[WARN] No images for class {cid} satisfying area_ratio >= {min_gt_area_ratio}")
            continue

        if len(valid_imgs) > max_images_per_class:
            chosen = list(rng.choice(valid_imgs, size=max_images_per_class, replace=False))
        else:
            chosen = valid_imgs

        print(f"[Select] class {cid} ({VOC_CLASSES[cid]}): {len(chosen)} images selected.")
        pairs.extend((cid, img_id) for img_id in chosen)

    print(f"[Select] Total image-class pairs: {len(pairs)}")
    return pairs


# =========================
# Precompute for one config
# =========================

def precompute_for_config(
    tag: str,
    hparam_name: str,
    hparam_value: float,
    pairs,
    voc_root: Path,
    sam2_cfg: str,
    sam2_ckpt: str,
    gdino_cfg: str,
    gdino_ckpt: str,
    output_root: Path,
    device: str = "cuda",
):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    out_root = output_root / tag
    print(f"\n=== Precompute for config '{tag}' ({hparam_name}={hparam_value}) ===")
    print(f"Output root: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading GroundingDINO...")
    gdino_model = build_groundingdino_model(gdino_cfg, gdino_ckpt, device=device)

    print("Loading SAM2...")
    sam2_predictor = build_sam2_predictor(sam2_cfg, sam2_ckpt, device=device)

    # 其它超参先固定死
    fixed_box_threshold = 0.25
    fixed_text_threshold = 0.20

    for idx, (cid, img_id) in enumerate(pairs):
        print(f"[{tag}] ({idx+1}/{len(pairs)}) class={cid}, img={img_id}")

        # 1) GT
        pil_img, mask_sem, img_path = load_voc_pair(voc_root, img_id)
        gt_mask = get_class_specific_mask(mask_sem, [cid])
        H, W = mask_sem.shape
        if gt_mask.sum() == 0:
            print("  [skip] GT empty")
            continue

        # 2) DINO 图像
        image_source, image_dino = gdino_load_image(img_path)
        h_src, w_src = image_source.shape[:2]

        if (h_src, w_src) != (H, W):
            gt_mask = cv2.resize(gt_mask.astype(np.uint8), (w_src, h_src),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
            H, W = h_src, w_src

        text_prompt = VOC_PROMPTS[cid] + " ."

        # 3) 注入超参
        box_threshold = fixed_box_threshold
        text_threshold = fixed_text_threshold

        if hparam_name == "box_threshold":
            box_threshold = hparam_value
        elif hparam_name == "text_threshold":
            text_threshold = hparam_value

        boxes_cxcywh, logits, phrases = gdino_predict(
            model=gdino_model,
            image=image_dino,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )

        if boxes_cxcywh.numel() == 0:
            print("  [warn] no boxes from GDINO, skip.")
            continue

        boxes_xyxy = boxes_cxcywh_to_xyxy_pixels(boxes_cxcywh, img_w=W, img_h=H)
        dino_logits = logits.cpu().numpy()

        # 4) SAM2
        image_rgb = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
        sam2_predictor.set_image(image_rgb)

        masks_list = []
        confs_list = []

        for box_xyxy, dino_logit, phrase in zip(boxes_xyxy, dino_logits, phrases):
            masks_np, sam_confs, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_xyxy[None, :],
                multimask_output=True,
                return_logits=False,
            )
            if masks_np.ndim != 3:
                continue
            for m_idx in range(masks_np.shape[0]):
                masks_list.append(masks_np[m_idx].astype(bool))
                confs_list.append(float(sam_confs[m_idx]))

        if not masks_list:
            print("  [warn] SAM produced no masks, skip.")
            continue

        masks_arr = np.stack(masks_list, axis=0)
        confs_arr = np.array(confs_list, dtype=np.float32)

        # 5) save npz
        class_dir = out_root / f"class_{cid}"
        class_dir.mkdir(parents=True, exist_ok=True)
        npz_path = class_dir / f"{img_id}.npz"

        np.savez_compressed(
            npz_path,
            class_id=np.int32(cid),
            image_id=np.bytes_(img_id),
            gt_mask=gt_mask.astype(bool),
            masks=masks_arr.astype(bool),
            sam_confs=confs_arr.astype(np.float32),
            hparam_name=np.bytes_(hparam_name),
            hparam_value=np.float32(hparam_value),
        )

    print(f"[{tag}] precompute done.")


# =========================
# 加载 & 比较
# =========================

def load_npz_for_config(output_root: Path, tag: str, class_ids):
    root = output_root / tag
    data = {}
    for cid in class_ids:
        class_dir = root / f"class_{cid}"
        if not class_dir.is_dir():
            continue
        for npz_path in sorted(class_dir.glob("*.npz")):
            d = np.load(npz_path, allow_pickle=True)
            gt_mask = d["gt_mask"].astype(bool)
            masks = d["masks"].astype(bool)
            sam_confs = d["sam_confs"].astype(float)

            img_id = d.get("image_id", npz_path.stem)
            if isinstance(img_id, np.ndarray):
                img_id = img_id.item()
            img_id = str(img_id)

            key = (int(cid), img_id)
            data[key] = (gt_mask, masks, sam_confs)
    print(f"[Load] config '{tag}': loaded {len(data)} samples.")
    return data


def compare_configs_by_score_and_iou(
    output_root: Path,
    hparam_name: str,
    tags,
    class_ids,
    q_border: float,
    summary_csv: Path,
):
    config_data = {tag: load_npz_for_config(output_root, tag, class_ids) for tag in tags}
    key_sets = [set(d.keys()) for d in config_data.values()]
    common_keys = set.intersection(*key_sets) if key_sets else set()
    print(f"[Compare] common samples across configs: {len(common_keys)}")

    if not common_keys:
        print("[Compare] No common samples; cannot compare.")
        return

    results = {}

    for tag, data in config_data.items():
        scores = []
        ious = []

        for key in sorted(common_keys):
            gt, masks, sam_confs = data[key]
            K, H, W = masks.shape

            best_idx = int(np.argmax(sam_confs))
            pred = masks[best_idx]

            score_list, _ = bench.compute_scores_new(
                [pred],
                W=W,
                H=H,
                q_border=q_border,
                sigmoid_isOn=bench.sigmoid_isOn,
            )
            s = float(score_list[0])
            iou = bench.compute_iou(pred, gt)

            scores.append(s)
            ious.append(iou)

        scores = np.array(scores, dtype=float)
        ious = np.array(ious, dtype=float)

        results[tag] = {
            "scores": scores,
            "ious": ious,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "mean_iou": float(np.mean(ious)),
            "std_iou": float(np.std(ious)),
            "n_samples": len(scores),
        }

    print("\n=== Comparison results on common samples ===")
    for tag in tags:
        r = results[tag]
        print(
            f"Config '{tag}': n={r['n_samples']}, "
            f"score_mean={r['mean_score']:.4f}±{r['std_score']:.4f}, "
            f"IoU_mean={r['mean_iou']:.4f}±{r['std_iou']:.4f}"
        )

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "hparam_name",
                "tag",
                "n_samples",
                "mean_score",
                "std_score",
                "mean_iou",
                "std_iou",
            ],
        )
        writer.writeheader()
        for tag in tags:
            r = results[tag]
            writer.writerow(
                {
                    "hparam_name": hparam_name,
                    "tag": tag,
                    "n_samples": r["n_samples"],
                    "mean_score": r["mean_score"],
                    "std_score": r["std_score"],
                    "mean_iou": r["mean_iou"],
                    "std_iou": r["std_iou"],
                }
            )
    print(f"[Compare] Summary CSV saved to {summary_csv}")


# =========================
# main
# =========================

def main():
    cfg = CONFIG

    voc_root = Path(cfg["VOC_ROOT"])
    sam2_cfg = cfg["SAM2_CFG"]
    sam2_ckpt = cfg["SAM2_CKPT"]
    gdino_cfg = cfg["GDINO_CFG"]
    gdino_ckpt = cfg["GDINO_CKPT"]

    class_ids = cfg["CLASS_IDS"]
    split = cfg["SPLIT"]
    max_imgs = cfg["MAX_IMAGES_PER_CLASS"]
    min_gt_ratio = cfg["MIN_GT_AREA_RATIO"]
    seed = cfg["SEED"]
    hparam_name = cfg["HPARAM_NAME"]
    hparam_values = cfg["HPARAM_VALUES"]
    output_root = Path(cfg["OUTPUT_ROOT"])
    device = cfg["DEVICE"]
    q_border = cfg["Q_BORDER"]

    output_root.mkdir(parents=True, exist_ok=True)

    print("=== Simple Hparam Sweep Experiment ===")
    print(f"  VOC_ROOT = {voc_root}")
    print(f"  CLASS_IDS = {class_ids}")
    print(f"  HPARAM = {hparam_name}, values = {hparam_values}")
    print(f"  OUTPUT_ROOT = {output_root}")

    # 1) 固定图像集合
    pairs = select_images_for_classes(
        voc_root=voc_root,
        class_ids=class_ids,
        split=split,
        min_gt_area_ratio=min_gt_ratio,
        max_images_per_class=max_imgs,
        seed=seed,
    )
    if not pairs:
        print("No (class_id,image_id) pairs selected. Exit.")
        return

    # 2) 每个超参数档位都跑一遍 precompute
    tags = []
    for val in hparam_values:
        tag = f"{hparam_name}_{val:.3f}".replace(".", "p")
        tags.append(tag)
        precompute_for_config(
            tag=tag,
            hparam_name=hparam_name,
            hparam_value=val,
            pairs=pairs,
            voc_root=voc_root,
            sam2_cfg=sam2_cfg,
            sam2_ckpt=sam2_ckpt,
            gdino_cfg=gdino_cfg,
            gdino_ckpt=gdino_ckpt,
            output_root=output_root,
            device=device,
        )

    # 3) 比较三套 config 的 SAM-Top1 score & IoU
    summary_csv = output_root / f"summary_{hparam_name}.csv"
    compare_configs_by_score_and_iou(
        output_root=output_root,
        hparam_name=hparam_name,
        tags=tags,
        class_ids=class_ids,
        q_border=q_border,
        summary_csv=summary_csv,
    )


if __name__ == "__main__":
    main()
