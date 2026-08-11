import os
import json
from pathlib import Path
import numpy as np
import cv2
import pycocotools.mask as mask_util

def _compute_distance_transform(h, w):
    """整图距边缘距离图（用于边缘分项 E）。"""
    border_mask = np.zeros((h, w), np.uint8)
    border_mask[1:-1, 1:-1] = 1
    return cv2.distanceTransform(border_mask, distanceType=cv2.DIST_L2, maskSize=5)

def _decode_bool_mask(segmentation_rle):
    """RLE -> bool(H,W)"""
    m = mask_util.decode(segmentation_rle)
    if m.ndim == 3:  # pycocotools 有时返回 (H,W,1)
        m = m[:, :, 0]
    return m.astype(bool)

def _recompute_scores_for_image(annotations, img_w, img_h,
                                W_AREA=0.2, W_CENTER=0.2, W_BORDER=0.6,
                                Q_BORDER=0.25, MIN_AREA_RATIO=0.01):
    """
    按【新算法】重算一张图所有 mask 的分数，并返回：
      - new_scores: [float]
      - terms: 每个mask的分项字典（A_norm, C, E, S_area, S_center, S_border）
      - best_idx, best_score
    """
    cx, cy = img_w / 2.0, img_h / 2.0
    img_area = float(img_w * img_h)
    dist_map = _compute_distance_transform(img_h, img_w)
    d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

    new_scores = []
    terms = []

    for ann in annotations:
        seg = ann["segmentation"]
        mask = _decode_bool_mask(seg)
        area = int(mask.sum())

        if area < MIN_AREA_RATIO * img_area or area == 0:
            new_scores.append(0.0)
            terms.append({
                "A_norm": 0.0, "C": 0.0, "E": 0.0,
                "S_area": 0.0, "S_center": 0.0, "S_border": 0.0
            })
            continue

        # 新面积项：mask像素 / 全图像素
        A_norm = area / img_area

        ys, xs = np.where(mask)
        mx, my = xs.mean(), ys.mean()
        D_prime = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(img_w, img_h))
        C = 1.0 - np.clip(D_prime, 0.0, 1.0)

        mask_distances = dist_map[mask]
        if mask_distances.size == 0:
            new_scores.append(0.0)
            terms.append({
                "A_norm": 0.0, "C": 0.0, "E": 0.0,
                "S_area": 0.0, "S_center": 0.0, "S_border": 0.0
            })
            continue

        q_val = float(np.quantile(mask_distances, Q_BORDER))
        B_prime = 1.0 - (q_val / d_max)         # 越贴边 B' 越大
        E = 1.0 - np.clip(B_prime, 0.0, 1.0)    # 越不贴边越高

        S_area   = W_AREA   * A_norm
        S_center = W_CENTER * C
        S_border = W_BORDER * E
        S = S_area + S_center + S_border

        new_scores.append(float(S))
        terms.append({
            "A_norm": float(A_norm),
            "C": float(C),
            "E": float(E),
            "S_area": float(S_area),
            "S_center": float(S_center),
            "S_border": float(S_border),
        })

    # 选最佳
    if new_scores:
        best_idx = int(np.argmax(new_scores))
        best_score = float(new_scores[best_idx])
    else:
        best_idx, best_score = -1, 0.0

    return new_scores, terms, best_idx, best_score


def rewrite_all_scores_in_folder(
    folder,
    W_AREA=0.2, W_CENTER=0.2, W_BORDER=0.6, Q_BORDER=0.25,
    MIN_AREA_RATIO=0.01,
    backup=True,
    file_suffix="_result.json"
):
    """
    用【新算法】批量重写文件夹内所有 *_result.json 的自定义分数字段：
      - 覆盖 annotations[*].score_custom
      - 覆盖 best_mask_index / best_score_custom
      - 为每个 mask 写入 score_terms_new（分项）
    若 backup=True，会生成同名 .bak.json 备份。
    """
    folder = Path(folder)
    files = sorted([p for p in folder.glob(f"*{file_suffix}") if p.is_file()])
    if not files:
        print(f"[rewrite] 没找到匹配的 JSON：{folder}\\*{file_suffix}")
        return

    print(f"[rewrite] 发现 {len(files)} 个 JSON，开始重写…")
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 读取图像尺寸（优先 json 中 img_width/img_height，否则用 segmentation.size）
            if "img_width" in data and "img_height" in data:
                W, H = int(data["img_width"]), int(data["img_height"])
            else:
                # 兜底：从第一条mask里取
                seg = data["annotations"][0]["segmentation"]
                # pycoco 的size通常是 [H, W]
                H, W = int(seg["size"][0]), int(seg["size"][1])

            anns = data.get("annotations", [])
            new_scores, terms, best_idx, best_score = _recompute_scores_for_image(
                anns, W, H,
                W_AREA=W_AREA, W_CENTER=W_CENTER, W_BORDER=W_BORDER,
                Q_BORDER=Q_BORDER, MIN_AREA_RATIO=MIN_AREA_RATIO
            )

            # 写回
            for ann, s, t in zip(anns, new_scores, terms):
                ann["score_custom"] = float(s)             # 覆盖旧字段
                ann["score_terms_new"] = t                 # 新增分项（便于UI展示）
                # 可选：保留 SAM 分数 ann["score_sam"] 不改

            # 同步顶层最佳
            data["best_mask_index"] = int(best_idx)
            data["best_score_custom"] = float(best_score)

            # （可选）记录这次使用的权重与参数，便于追踪
            data["custom_score_config"] = {
                "W_AREA": W_AREA,
                "W_CENTER": W_CENTER,
                "W_BORDER": W_BORDER,
                "Q_BORDER": Q_BORDER,
                "MIN_AREA_RATIO": MIN_AREA_RATIO,
                "area_term": "mask_pixels / image_pixels"  # 明确声明新算法
            }

            # 备份
            if backup:
                bak = fp.with_suffix(fp.suffix + ".bak")
                with open(bak, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                # 再写回原文件
                with open(fp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                with open(fp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"[rewrite] ✅ {fp.name} 已更新（best_idx={best_idx}, best_score={best_score:.4f})")

        except Exception as e:
            print(f"[rewrite] ❌ {fp.name} 失败：{e}")

    print("[rewrite] 全部完成。")


def compare_score_logic(json_path):
    from AdjustWeight import compute_custom_scores_with_terms
    import json, pycocotools.mask as mask_util

    with open(json_path, "r") as f:
        data = json.load(f)
    anns = data["annotations"]
    h, w = data["img_height"], data["img_width"]
    masks_bool = [mask_util.decode(a["segmentation"]).astype(bool) for a in anns]
    ui_scores = [s[0] for s in compute_custom_scores_with_terms(masks_bool, w, h)]
    json_scores = [a["score_custom"] for a in anns]
    diffs = np.abs(np.array(ui_scores) - np.array(json_scores))
    print(f"平均误差={diffs.mean():.6f}, 最大误差={diffs.max():.6f}")

if __name__ == "__main__":
    # 示例用法
    # rewrite_all_scores_in_folder(
    #     folder="outputs/AdjustWeightData",
    #     W_AREA=0.2, W_CENTER=0.2, W_BORDER=0.6, Q_BORDER=0.25,
    #     MIN_AREA_RATIO=0.01,
    #     backup=True  # 留备份，安全
    # )
    compare_score_logic("outputs/AllMasks_v2_score_record/img_0001_result.json")