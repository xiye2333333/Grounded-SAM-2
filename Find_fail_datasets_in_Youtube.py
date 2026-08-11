import os
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# =========================================================
# Config
# =========================================================
ANNOTATIONS_ROOT = r"D:\uwb thesis\RelatedData\train\train\Annotations"
OUTPUT_JSON = r"D:\uwb thesis\RelatedData\train\train\annotation_analysis_streaming.json"

# 中心/边缘判定阈值（质心到图像中心的归一化距离）
CENTER_THRESHOLD = 0.35
EDGE_THRESHOLD = 0.70

# 后期比前期至少增加多少，才算“从中间移动到边缘”
MOVEMENT_MIN_INCREASE = 0.20

# 前多少比例帧作为 early，后多少比例帧作为 late
PHASE_RATIO = 0.25

# 是否保存每帧细节
SAVE_PER_FRAME_DETAILS = False


# =========================================================
# Basic Utilities
# =========================================================
def read_mask_unchanged(path: str):
    """
    以原始格式读取 mask。
    可能得到：
    - H x W          (灰度/索引图)
    - H x W x 3      (RGB/BGR)
    - H x W x 4      (带 alpha)
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def foreground_binary(mask: np.ndarray) -> np.ndarray:
    """
    将 mask 转成二值前景：
    - 非零 => 前景
    - 零   => 背景
    返回 uint8, 值为 0/255
    """
    if mask.ndim == 2:
        fg = (mask != 0)
    else:
        fg = np.any(mask != 0, axis=2)

    return fg.astype(np.uint8) * 255


def get_single_foreground_signature(mask: np.ndarray):
    """
    快速判断这一帧是否只有一种非零前景“值/颜色”。
    返回:
        (is_single, signature)

    signature:
      - 灰度/索引图时: int
      - 彩色图时: tuple
      - 没有前景时: None
    """
    if mask.ndim == 2:
        fg = mask[mask != 0]
        if fg.size == 0:
            return False, None

        first = fg[0]
        if np.all(fg == first):
            return True, int(first)

        vals = np.unique(fg)
        if len(vals) == 1:
            return True, int(vals[0])
        return False, None

    else:
        fg_mask = np.any(mask != 0, axis=2)
        coords = np.where(fg_mask)
        if len(coords[0]) == 0:
            return False, None

        fg_pixels = mask[coords]
        first = fg_pixels[0]

        if np.all(fg_pixels == first):
            return True, tuple(int(x) for x in first.tolist())

        vals = np.unique(fg_pixels, axis=0)
        if len(vals) == 1:
            return True, tuple(int(x) for x in vals[0].tolist())
        return False, None


def compute_centroid_and_norm_dist(binary_mask: np.ndarray):
    """
    计算前景质心，以及质心到图像中心的归一化距离。
    返回:
        centroid: (cx, cy) or None
        norm_dist: float or None
    """
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return None, None

    cx = float(xs.mean())
    cy = float(ys.mean())

    h, w = binary_mask.shape[:2]
    img_cx = (w - 1) / 2.0
    img_cy = (h - 1) / 2.0

    dist = np.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2)
    max_dist = np.sqrt(img_cx ** 2 + img_cy ** 2)

    if max_dist < 1e-8:
        norm_dist = 0.0
    else:
        norm_dist = float(dist / max_dist)

    return (cx, cy), norm_dist


def mask_has_hole_fast(binary_mask: np.ndarray) -> bool:
    """
    用 contour hierarchy 判断是否有内部洞。
    """
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)

    contours, hierarchy = cv2.findContours(
        binary_mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return False

    hierarchy = hierarchy[0]
    # [next, prev, child, parent]
    for h in hierarchy:
        if h[3] != -1:
            return True
    return False


def is_valid_image_file(p: Path) -> bool:
    return p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


# =========================================================
# Stage 1: Quick Single-Object Screening
# =========================================================
def quick_check_single_object_video(video_dir: Path):
    """
    快速检查一个视频文件夹是否为“单目标视频”：
    - 每一帧都必须只有一种非零前景值/颜色
    - 整个视频里前景 signature 必须一致
    """
    image_paths = sorted([p for p in video_dir.iterdir() if p.is_file() and is_valid_image_file(p)])

    if len(image_paths) == 0:
        return {
            "ok": False,
            "num_frames": 0,
            "global_signature": None,
            "reason": "empty_folder"
        }

    global_signature = None

    for img_path in image_paths:
        mask = read_mask_unchanged(str(img_path))
        is_single, sig = get_single_foreground_signature(mask)

        if not is_single:
            return {
                "ok": False,
                "num_frames": len(image_paths),
                "global_signature": None,
                "reason": f"frame_not_single_object: {img_path.name}"
            }

        if global_signature is None:
            global_signature = sig
        elif sig != global_signature:
            return {
                "ok": False,
                "num_frames": len(image_paths),
                "global_signature": None,
                "reason": f"inconsistent_foreground_signature: {img_path.name}"
            }

    return {
        "ok": True,
        "num_frames": len(image_paths),
        "global_signature": global_signature,
        "reason": None
    }


# =========================================================
# Stage 2: Detailed Analysis for Qualified Videos
# =========================================================
def analyze_single_object_video(video_dir: Path):
    image_paths = sorted([p for p in video_dir.iterdir() if p.is_file() and is_valid_image_file(p)])

    distances = []
    hole_frames = []
    per_frame_details = []

    for img_path in image_paths:
        mask = read_mask_unchanged(str(img_path))
        binary = foreground_binary(mask)

        centroid, norm_dist = compute_centroid_and_norm_dist(binary)
        has_hole = mask_has_hole_fast(binary)

        if norm_dist is not None:
            distances.append(norm_dist)

        if has_hole:
            hole_frames.append(img_path.name)

        if SAVE_PER_FRAME_DETAILS:
            per_frame_details.append({
                "frame": img_path.name,
                "centroid": None if centroid is None else [centroid[0], centroid[1]],
                "center_distance": norm_dist,
                "has_hole": has_hole
            })

    movement_stats = {}
    moves_center_to_edge = False

    if len(distances) >= 4:
        n = len(distances)
        k = max(1, int(round(n * PHASE_RATIO)))

        early = distances[:k]
        late = distances[-k:]

        early_mean = float(np.mean(early))
        late_mean = float(np.mean(late))
        increase = late_mean - early_mean

        starts_center = early_mean <= CENTER_THRESHOLD
        ends_near_edge = late_mean >= EDGE_THRESHOLD
        enough_increase = increase >= MOVEMENT_MIN_INCREASE

        moves_center_to_edge = starts_center and ends_near_edge and enough_increase

        movement_stats = {
            "num_frames_used": n,
            "early_mean_center_distance": early_mean,
            "late_mean_center_distance": late_mean,
            "increase": increase,
            "starts_center": starts_center,
            "ends_near_edge": ends_near_edge,
            "enough_increase": enough_increase
        }
    else:
        movement_stats = {
            "num_frames_used": len(distances),
            "reason": "too_few_valid_frames_for_motion_analysis"
        }

    return {
        "contains_hole": len(hole_frames) > 0,
        "hole_frames": hole_frames,
        "moves_from_center_to_edge": moves_center_to_edge,
        "movement_stats": movement_stats,
        "per_frame_details": per_frame_details if SAVE_PER_FRAME_DETAILS else None
    }


# =========================================================
# Streaming JSON Helpers
# =========================================================
def write_json_array_header(path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("[\n")


def append_json_object(path: str, obj: dict, is_last: bool):
    with open(path, "a", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        if not is_last:
            f.write(",\n")
        else:
            f.write("\n")


def write_json_array_footer(path: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write("]\n")


# =========================================================
# Main
# =========================================================
def main():
    root = Path(ANNOTATIONS_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")

    video_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    total_videos = len(video_dirs)

    print(f"Found {total_videos} video folders.")
    print(f"Streaming results to: {OUTPUT_JSON}")

    # 实时累积统计
    single_object_videos = []
    center_to_edge_videos = []
    hole_videos = []

    # 初始化输出 JSON（以数组形式逐条写入）
    write_json_array_header(OUTPUT_JSON)

    for idx, video_dir in enumerate(tqdm(video_dirs, desc="Scanning videos")):
        # -------- Stage 1: quick screening --------
        quick = quick_check_single_object_video(video_dir)

        result = {
            "video": video_dir.name,
            "num_frames": quick["num_frames"],
            "is_single_object_video": quick["ok"],
            "foreground_signature": quick["global_signature"],
            "quick_check_reason": quick["reason"],
            "moves_from_center_to_edge": False,
            "contains_hole": False,
            "hole_frames": [],
            "movement_stats": None
        }

        # -------- Stage 2: only for qualified videos --------
        if quick["ok"]:
            single_object_videos.append(video_dir.name)
            print(f"[MATCH single] {video_dir.name}")

            detail = analyze_single_object_video(video_dir)
            result["moves_from_center_to_edge"] = detail["moves_from_center_to_edge"]
            result["contains_hole"] = detail["contains_hole"]
            result["hole_frames"] = detail["hole_frames"]
            result["movement_stats"] = detail["movement_stats"]

            if SAVE_PER_FRAME_DETAILS:
                result["per_frame_details"] = detail["per_frame_details"]

            if detail["moves_from_center_to_edge"]:
                center_to_edge_videos.append(video_dir.name)
                print(f"[MATCH center_to_edge] {video_dir.name}")

            if detail["contains_hole"]:
                hole_videos.append(video_dir.name)
                print(f"[MATCH hole] {video_dir.name}")

        # -------- 实时写入当前视频结果 --------
        is_last = (idx == total_videos - 1)
        append_json_object(OUTPUT_JSON, result, is_last=is_last)

    # 补上 JSON 数组结束符
    write_json_array_footer(OUTPUT_JSON)

    print("\n================ FINAL SUMMARY ================")
    print(f"Total video folders: {total_videos}")
    print(f"Single-object videos: {len(single_object_videos)}")
    print(f"Single-object videos moving center -> edge: {len(center_to_edge_videos)}")
    print(f"Single-object videos containing holes: {len(hole_videos)}")

    print("\nSingle-object video names:")
    for name in single_object_videos:
        print(f"  {name}")

    print("\nCenter-to-edge video names:")
    for name in center_to_edge_videos:
        print(f"  {name}")

    print("\nHole video names:")
    for name in hole_videos:
        print(f"  {name}")

    print(f"\nSaved streaming results to:\n{OUTPUT_JSON}")


if __name__ == "__main__":
    main()