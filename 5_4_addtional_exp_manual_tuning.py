import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


# =========================================================
# Global config
# =========================================================

DEFAULT_ROOT = r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results"
EPS = 1e-8


# =========================================================
# RLE / mask utilities
# =========================================================

def decode_rle(segmentation_rle):
    """
    Decode COCO-style RLE from JSON.

    Supports:
    1. compressed RLE: counts is string
    2. uncompressed RLE: counts is list

    Returns:
        mask_bool: np.ndarray[bool], shape = (H, W)
    """
    if segmentation_rle is None:
        raise ValueError("segmentation_rle is None.")

    size = segmentation_rle.get("size", None)
    counts = segmentation_rle.get("counts", None)

    if size is None or counts is None:
        raise ValueError("Invalid RLE: missing size or counts.")

    h, w = int(size[0]), int(size[1])

    # Compressed COCO RLE, usually counts is a string.
    if isinstance(counts, str):
        if mask_utils is None:
            raise ImportError(
                "pycocotools is required to decode compressed RLE counts. "
                "Install with: pip install pycocotools"
            )
        rle = {
            "size": [h, w],
            "counts": counts.encode("utf-8")
        }
        mask = mask_utils.decode(rle)
        return mask.astype(bool)

    # Some JSON libraries may load compressed bytes-like content oddly.
    if isinstance(counts, bytes):
        if mask_utils is None:
            raise ImportError(
                "pycocotools is required to decode compressed RLE counts."
            )
        rle = {
            "size": [h, w],
            "counts": counts
        }
        mask = mask_utils.decode(rle)
        return mask.astype(bool)

    # Uncompressed COCO RLE, counts is list.
    if isinstance(counts, list):
        flat = []
        value = 0
        for run_len in counts:
            flat.extend([value] * int(run_len))
            value = 1 - value

        arr = np.array(flat, dtype=np.uint8)

        expected = h * w
        if arr.size < expected:
            arr = np.pad(arr, (0, expected - arr.size), mode="constant")
        elif arr.size > expected:
            arr = arr[:expected]

        # COCO RLE is column-major.
        mask = arr.reshape((w, h)).T
        return mask.astype(bool)

    raise TypeError(f"Unsupported RLE counts type: {type(counts)}")


def read_mask_png(path):
    """
    Read a png mask and convert it to bool.
    Assumes non-zero pixels are foreground.
    """
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return arr > 0


def mask_to_pil(mask_bool):
    """
    Convert bool mask to grayscale PIL image.
    """
    arr = (mask_bool.astype(np.uint8) * 255)
    return Image.fromarray(arr)


def overlay_mask_on_image(image_pil, mask_bool, color=(255, 0, 0), alpha=0.45):
    """
    Overlay a binary mask on raw image.
    """
    image = np.array(image_pil.convert("RGB")).astype(np.float32)
    mask = mask_bool.astype(bool)

    overlay = image.copy()
    overlay[mask] = (
        (1.0 - alpha) * overlay[mask]
        + alpha * np.array(color, dtype=np.float32)
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def compute_iou(mask_a, mask_b):
    """
    IoU between two bool masks.
    """
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0
    return float(inter / union)


# =========================================================
# Feature functions
#   [A, C, E, Sil]
# =========================================================

def compute_distance_transform(h, w):
    border_mask = np.zeros((h, w), np.uint8)
    border_mask[1:-1, 1:-1] = 1
    return cv2.distanceTransform(
        border_mask,
        distanceType=cv2.DIST_L2,
        maskSize=5
    )


def area_term_parabola(x: float, d: float, eps: float = 1e-6) -> float:
    d = float(np.clip(d, eps, 1.0))
    x = float(np.clip(x, 0.0, 1.0))
    val = 1.0 - ((x - d) / d) ** 2
    return float(np.clip(val, 0.0, 1.0))


def compute_silhouette_score_v2(mask: np.ndarray) -> float:
    mask_u8 = (mask.astype(np.uint8) > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    if area <= 0:
        return 0.0

    hull = cv2.convexHull(largest)
    hull_area = float(cv2.contourArea(hull))
    solidity = 0.0 if hull_area <= 0 else float(
        np.clip(area / hull_area, 0.0, 1.0)
    )

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8
    )
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
                fragmentation = 1.0 - (
                    largest_area / float(np.sum(large_areas))
                )

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

    # A: area prior
    A_raw = area_px / img_area
    A = area_term_parabola(A_raw, t_area)

    # C: center prior
    ys, xs = np.where(mask_bool)
    mx, my = xs.mean(), ys.mean()
    Dp = np.hypot(mx - cx, my - cy) / (0.5 * np.hypot(W, H))
    C = 1.0 - float(np.clip(Dp, 0.0, 1.0))

    # E: border-distance / interior support term
    q = float(np.quantile(dist_map[mask_bool], q_border))
    E = float(np.clip(q / d_max, 0.0, 1.0))

    # Sil: silhouette structure
    Sil = compute_silhouette_score_v2(mask_bool)

    return np.array([A, C, E, Sil], dtype=float)


def compute_score(features, weights):
    """
    features: [A, C, E, Sil]
    weights: normalized [wA, wC, wE, wSil]
    """
    return float(np.dot(features, weights))


# =========================================================
# Dataset loading
# =========================================================

def list_datasets(root):
    root = Path(root)
    if not root.exists():
        return []

    return sorted([
        p.name for p in root.iterdir()
        if p.is_dir()
    ])


def list_frame_ids(dataset_dir):
    gt_dir = dataset_dir / "gt_json"
    if not gt_dir.exists():
        return []

    frame_ids = []
    for p in gt_dir.glob("*_gt.json"):
        m = re.match(r"(\d+)_gt\.json$", p.name)
        if m:
            frame_ids.append(m.group(1))

    return sorted(frame_ids)


@st.cache_data(show_spinner=False)
def load_gt_json(gt_json_path_str):
    with open(gt_json_path_str, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_mask = decode_rle(data["segmentation_rle"])
    image_path = data["image_path"]
    width = int(data["width"])
    height = int(data["height"])

    return {
        "image_path": image_path,
        "width": width,
        "height": height,
        "gt_mask": gt_mask,
    }


@st.cache_data(show_spinner=False)
def load_result_json(result_json_path_str):
    with open(result_json_path_str, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


@st.cache_data(show_spinner=False)
def load_raw_image(image_path_str):
    return Image.open(image_path_str).convert("RGB")


def find_sam_choice_path(dataset_dir, frame_id):
    """
    selected_by_sam filename may be:
    00000_sam_best.png
    00000_sam_best_png.png
    or similar.
    This function tries several robust patterns.
    """
    sam_dir = dataset_dir / "selected_by_sam"
    if not sam_dir.exists():
        return None

    candidates = []

    exact_names = [
        f"{frame_id}_sam_best.png",
        f"{frame_id}_sam_best_png.png",
        f"{frame_id}_sam_best_png",
    ]

    for name in exact_names:
        p = sam_dir / name
        if p.exists():
            return p

    candidates.extend(sam_dir.glob(f"{frame_id}*sam*best*.png"))
    candidates.extend(sam_dir.glob(f"{frame_id}*.png"))

    if len(candidates) > 0:
        return sorted(candidates)[0]

    return None


def find_candidate_mask_path(dataset_dir, frame_id, ann_id):
    """
    Find candidate mask png from masks folder.

    Expected:
        00000_m0000.png
        00000_m0001.png

    ann_id is usually consistent with mXXXX.
    """
    mask_dir = dataset_dir / "masks"
    if not mask_dir.exists():
        return None

    exact = mask_dir / f"{frame_id}_m{ann_id:04d}.png"
    if exact.exists():
        return exact

    candidates = list(mask_dir.glob(f"{frame_id}_m{ann_id:04d}*.png"))
    if candidates:
        return sorted(candidates)[0]

    return None


def load_candidate_masks_from_json(result_json):
    """
    Decode all candidate masks from result json.

    Returns:
        list of dict:
        {
            "id": int,
            "box_id": int,
            "rank_in_box": int,
            "sam_score": float,
            "mask": np.ndarray[bool]
        }
    """
    anns = result_json.get("annotations", [])
    candidates = []

    for idx, ann in enumerate(anns):
        ann_id = int(ann.get("id", idx))
        mask = decode_rle(ann["segmentation"])

        candidates.append({
            "id": ann_id,
            "box_id": int(ann.get("box_id", -1)),
            "rank_in_box": int(ann.get("rank_in_box", -1)),
            "sam_score": float(ann.get("sam_score", np.nan)),
            "mask": mask,
        })

    return candidates

# =========================================================
# Sequence-level evaluation under current parameters
# =========================================================

def evaluate_sequence_mean_iou(
    dataset_dir,
    frame_ids,
    weights,
    t_area,
    q_border,
):
    """
    Evaluate the current scoring function on the entire selected sequence.

    For each frame:
        1. Load GT mask.
        2. Load all SAM candidate masks from result json.
        3. Compute score for each candidate under current parameters.
        4. Select the score-best candidate.
        5. Compute IoU between selected candidate and GT.

    Returns:
        mean_iou: float
        result_df: pd.DataFrame
    """
    records = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    total = len(frame_ids)

    for idx, fid in enumerate(frame_ids):
        status_text.write(f"Evaluating frame {fid} ({idx + 1}/{total})...")

        gt_json_path = dataset_dir / "gt_json" / f"{fid}_gt.json"
        result_json_path = dataset_dir / "json" / f"{fid}_result.json"

        if not gt_json_path.exists() or not result_json_path.exists():
            records.append({
                "frame_id": fid,
                "selected_id": None,
                "score": np.nan,
                "IoU_vs_GT": np.nan,
                "num_candidates": 0,
                "note": "missing gt_json or result_json"
            })
            progress_bar.progress((idx + 1) / total)
            continue

        try:
            gt_data_local = load_gt_json(str(gt_json_path))
            result_json_local = load_result_json(str(result_json_path))

            gt_mask_local = gt_data_local["gt_mask"]
            H_local, W_local = gt_mask_local.shape

            dist_map_local = compute_distance_transform(H_local, W_local)
            d_max_local = float(dist_map_local.max()) if dist_map_local.max() > 0 else 1.0

            candidates_local = load_candidate_masks_from_json(result_json_local)

            best_local = None

            for cand in candidates_local:
                mask = cand["mask"]

                if mask.shape != gt_mask_local.shape:
                    continue

                feats = mask_features(
                    mask_bool=mask,
                    W=W_local,
                    H=H_local,
                    dist_map=dist_map_local,
                    d_max=d_max_local,
                    q_border=q_border,
                    t_area=t_area,
                )

                score = compute_score(feats, weights)
                iou = compute_iou(mask, gt_mask_local)

                if best_local is None or score > best_local["score"]:
                    best_local = {
                        "selected_id": int(cand["id"]),
                        "score": score,
                        "IoU_vs_GT": iou,
                        "A": feats[0],
                        "C": feats[1],
                        "E": feats[2],
                        "Sil": feats[3],
                    }

            if best_local is None:
                records.append({
                    "frame_id": fid,
                    "selected_id": None,
                    "score": np.nan,
                    "IoU_vs_GT": np.nan,
                    "num_candidates": len(candidates_local),
                    "note": "no valid candidate"
                })
            else:
                records.append({
                    "frame_id": fid,
                    "selected_id": best_local["selected_id"],
                    "score": best_local["score"],
                    "IoU_vs_GT": best_local["IoU_vs_GT"],
                    "A": best_local["A"],
                    "C": best_local["C"],
                    "E": best_local["E"],
                    "Sil": best_local["Sil"],
                    "num_candidates": len(candidates_local),
                    "note": "ok"
                })

        except Exception as e:
            records.append({
                "frame_id": fid,
                "selected_id": None,
                "score": np.nan,
                "IoU_vs_GT": np.nan,
                "num_candidates": 0,
                "note": f"error: {e}"
            })

        progress_bar.progress((idx + 1) / total)

    status_text.empty()
    progress_bar.empty()

    result_df = pd.DataFrame(records)
    mean_iou = float(result_df["IoU_vs_GT"].mean(skipna=True))

    return mean_iou, result_df
# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(
    page_title="Manual Parameter Tuning for SAM Mask Selection",
    layout="wide"
)

st.title("Manual Parameter Tuning for SAM Mask Selection")
st.caption(
    "Use partial GT samples to inspect SAM errors, adjust feature weights, "
    "and observe the selected mask under the current scoring function."
)

with st.sidebar:
    st.header("Dataset")

    root_str = st.text_input(
        "SAM_results root",
        value=DEFAULT_ROOT
    )

    root = Path(root_str)
    datasets = list_datasets(root_str)

    if not datasets:
        st.error("No dataset folders found. Please check the SAM_results root path.")
        st.stop()

    dataset_name = st.selectbox(
        "Target dataset",
        options=datasets,
        index=datasets.index("breakdance") if "breakdance" in datasets else 0
    )

    dataset_dir = root / dataset_name
    frame_ids = list_frame_ids(dataset_dir)

    if not frame_ids:
        st.error("No gt_json files found in this dataset.")
        st.stop()

    frame_id = st.selectbox(
        "Frame",
        options=frame_ids,
        index=0
    )

    st.divider()

    st.header("Scoring weights")

    wA = st.slider("w_A: area", 0.0, 1.0, 0.15, 0.01)
    wC = st.slider("w_C: center", 0.0, 1.0, 0.2, 0.01)
    wE = st.slider("w_E: border / interior support", 0.0, 1.0, 0.40, 0.01)
    wSil = st.slider("w_Sil: silhouette", 0.0, 1.0, 0.25, 0.01)

    weights_raw = np.array([wA, wC, wE, wSil], dtype=float)
    weights = normalize_weights(weights_raw)

    st.write("Normalized weights")
    st.dataframe(
        pd.DataFrame(
            [{
                "A": weights[0],
                "C": weights[1],
                "E": weights[2],
                "Sil": weights[3],
            }]
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    with st.expander("Advanced feature parameters", expanded=False):
        t_area = st.slider(
            "t_area: expected area ratio",
            0.001,
            1.0,
            0.20,
            0.001,
            help="Area term reaches its maximum when mask area ratio is near t_area."
        )

        q_border = st.slider(
            "q_border: quantile for border-distance term",
            0.0,
            1.0,
            0.20,
            0.01,
            help="Higher values emphasize stronger interior support farther from image border."
        )

    show_overlay = st.checkbox("Show masks as overlays on raw image", value=True)


# =========================================================
# Load selected frame
# =========================================================

gt_json_path = dataset_dir / "gt_json" / f"{frame_id}_gt.json"
result_json_path = dataset_dir / "json" / f"{frame_id}_result.json"

if not gt_json_path.exists():
    st.error(f"Missing GT json: {gt_json_path}")
    st.stop()

if not result_json_path.exists():
    st.error(f"Missing result json: {result_json_path}")
    st.stop()

gt_data = load_gt_json(str(gt_json_path))
result_json = load_result_json(str(result_json_path))

raw_image = load_raw_image(gt_data["image_path"])
gt_mask = gt_data["gt_mask"]

H, W = gt_mask.shape
dist_map = compute_distance_transform(H, W)
d_max = float(dist_map.max()) if dist_map.max() > 0 else 1.0

candidates = load_candidate_masks_from_json(result_json)

if not candidates:
    st.error("No candidate masks found in result json.")
    st.stop()


# =========================================================
# SAM choice
# =========================================================

sam_choice_path = find_sam_choice_path(dataset_dir, frame_id)

if sam_choice_path is not None:
    sam_choice_mask = read_mask_png(sam_choice_path)
    sam_iou = compute_iou(sam_choice_mask, gt_mask)
else:
    sam_choice_mask = None
    sam_iou = None


# =========================================================
# Compute current score for every candidate
# =========================================================

rows = []
best = None

for cand in candidates:
    mask = cand["mask"]

    if mask.shape != gt_mask.shape:
        st.warning(
            f"Candidate id={cand['id']} has shape {mask.shape}, "
            f"but GT has shape {gt_mask.shape}. Skipped."
        )
        continue

    feats = mask_features(
        mask_bool=mask,
        W=W,
        H=H,
        dist_map=dist_map,
        d_max=d_max,
        q_border=q_border,
        t_area=t_area,
    )

    score = compute_score(feats, weights)
    iou = compute_iou(mask, gt_mask)

    weighted_terms = feats * weights

    row = {
        "id": cand["id"],
        "box_id": cand["box_id"],
        "rank_in_box": cand["rank_in_box"],
        "sam_score": cand["sam_score"],
        "score": score,
        "IoU_vs_GT": iou,
        "A": feats[0],
        "C": feats[1],
        "E": feats[2],
        "Sil": feats[3],
        "wA*A": weighted_terms[0],
        "wC*C": weighted_terms[1],
        "wE*E": weighted_terms[2],
        "wSil*Sil": weighted_terms[3],
        "mask": mask,
    }

    rows.append(row)

    if best is None or score > best["score"]:
        best = row

if best is None:
    st.error("No valid candidate masks could be scored.")
    st.stop()

# =========================================================
# Compute current score for SAM's selected mask
# =========================================================

sam_choice_score = None
sam_choice_features = None
sam_choice_weighted_terms = None

if sam_choice_mask is not None:
    if sam_choice_mask.shape == gt_mask.shape:
        sam_choice_features = mask_features(
            mask_bool=sam_choice_mask,
            W=W,
            H=H,
            dist_map=dist_map,
            d_max=d_max,
            q_border=q_border,
            t_area=t_area,
        )
        sam_choice_score = compute_score(sam_choice_features, weights)
        sam_choice_weighted_terms = sam_choice_features * weights
    else:
        st.warning(
            f"SAM choice mask has shape {sam_choice_mask.shape}, "
            f"but GT has shape {gt_mask.shape}. SAM score is not computed."
        )

df = pd.DataFrame([
    {k: v for k, v in row.items() if k != "mask"}
    for row in rows
]).sort_values("score", ascending=False)


# =========================================================
# Top summary
# =========================================================

st.subheader(f"Dataset: {dataset_name} | Frame: {frame_id}")

summary_cols = st.columns(4)

with summary_cols[0]:
    st.metric("Number of candidates", len(rows))

with summary_cols[1]:
    if sam_iou is not None:
        st.metric("SAM choice IoU", f"{sam_iou:.4f}")
    else:
        st.metric("SAM choice IoU", "N/A")

with summary_cols[2]:
    st.metric("Current score-best id", int(best["id"]))

with summary_cols[3]:
    st.metric("Current score-best IoU", f"{best['IoU_vs_GT']:.4f}")

# =========================================================
# Sequence-level mean IoU under current parameters
# =========================================================

st.subheader("Sequence-level performance under current parameters")

st.write(
    "This evaluates all frames in the current sequence. "
    "For each frame, the mask with the highest current score is selected, "
    "and its IoU against GT is computed."
)

eval_col1, eval_col2 = st.columns([1, 3])

with eval_col1:
    run_sequence_eval = st.button(
        "Compute sequence mean IoU",
        type="primary"
    )

with eval_col2:
    st.write(
        f"Current sequence: `{dataset_name}` | "
        f"Number of frames: `{len(frame_ids)}`"
    )

if run_sequence_eval:
    sequence_mean_iou, sequence_eval_df = evaluate_sequence_mean_iou(
        dataset_dir=dataset_dir,
        frame_ids=frame_ids,
        weights=weights,
        t_area=t_area,
        q_border=q_border,
    )

    st.metric(
        "Current-parameter sequence mean IoU",
        f"{sequence_mean_iou:.4f}"
    )

    st.dataframe(
        sequence_eval_df,
        use_container_width=True,
        hide_index=True
    )

    csv = sequence_eval_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download sequence evaluation CSV",
        data=csv,
        file_name=f"{dataset_name}_current_parameter_sequence_eval.csv",
        mime="text/csv"
    )
# =========================================================
# Image display
# =========================================================

st.subheader("Visual comparison")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Raw image**")
    st.image(raw_image, use_container_width=True)

with col2:
    st.markdown("**GT**")
    if show_overlay:
        st.image(
            overlay_mask_on_image(raw_image, gt_mask, color=(0, 255, 0), alpha=0.45),
            use_container_width=True
        )
    else:
        st.image(mask_to_pil(gt_mask), use_container_width=True)

with col3:
    st.markdown("**SAM's choice**")
    if sam_choice_mask is not None:
        if show_overlay:
            st.image(
                overlay_mask_on_image(
                    raw_image,
                    sam_choice_mask,
                    color=(255, 0, 0),
                    alpha=0.45
                ),
                use_container_width=True
            )
        else:
            st.image(mask_to_pil(sam_choice_mask), use_container_width=True)

        st.write(f"IoU vs GT: `{sam_iou:.4f}`")

        if sam_choice_score is not None:
            st.write(f"Current score: `{sam_choice_score:.4f}`")

            with st.expander("SAM choice score composition", expanded=False):
                sam_comp_df = pd.DataFrame([
                    {
                        "Term": "A: area",
                        "Feature value": sam_choice_features[0],
                        "Normalized weight": weights[0],
                        "Weighted contribution": sam_choice_weighted_terms[0],
                    },
                    {
                        "Term": "C: center",
                        "Feature value": sam_choice_features[1],
                        "Normalized weight": weights[1],
                        "Weighted contribution": sam_choice_weighted_terms[1],
                    },
                    {
                        "Term": "E: border / interior support",
                        "Feature value": sam_choice_features[2],
                        "Normalized weight": weights[2],
                        "Weighted contribution": sam_choice_weighted_terms[2],
                    },
                    {
                        "Term": "Sil: silhouette",
                        "Feature value": sam_choice_features[3],
                        "Normalized weight": weights[3],
                        "Weighted contribution": sam_choice_weighted_terms[3],
                    },
                ])

                st.dataframe(
                    sam_comp_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.write(
                    f"Final score = "
                    f"{sam_choice_weighted_terms[0]:.4f} + "
                    f"{sam_choice_weighted_terms[1]:.4f} + "
                    f"{sam_choice_weighted_terms[2]:.4f} + "
                    f"{sam_choice_weighted_terms[3]:.4f} "
                    f"= **{sam_choice_score:.4f}**"
                )
        else:
            st.write("Current score: `N/A`")

        st.write(f"Path: `{sam_choice_path}`")
    else:
        st.warning("SAM selected mask not found.")

with col4:
    st.markdown("**Current score-best choice**")
    best_mask = best["mask"]

    candidate_png_path = find_candidate_mask_path(dataset_dir, frame_id, int(best["id"]))

    if show_overlay:
        st.image(
            overlay_mask_on_image(raw_image, best_mask, color=(0, 0, 255), alpha=0.45),
            use_container_width=True
        )
    else:
        if candidate_png_path is not None:
            st.image(str(candidate_png_path), use_container_width=True)
        else:
            st.image(mask_to_pil(best_mask), use_container_width=True)

    st.write(f"Candidate id: `{int(best['id'])}`")
    st.write(f"Score: `{best['score']:.4f}`")
    st.write(f"IoU vs GT: `{best['IoU_vs_GT']:.4f}`")

    if candidate_png_path is not None:
        st.write(f"Path: `{candidate_png_path}`")
    else:
        st.write("Candidate png not found; displayed decoded RLE mask instead.")


# =========================================================
# Score composition
# =========================================================

st.subheader("Score composition of current selected mask")

comp_df = pd.DataFrame([
    {
        "Term": "A: area",
        "Feature value": best["A"],
        "Normalized weight": weights[0],
        "Weighted contribution": best["wA*A"],
    },
    {
        "Term": "C: center",
        "Feature value": best["C"],
        "Normalized weight": weights[1],
        "Weighted contribution": best["wC*C"],
    },
    {
        "Term": "E: border / interior support",
        "Feature value": best["E"],
        "Normalized weight": weights[2],
        "Weighted contribution": best["wE*E"],
    },
    {
        "Term": "Sil: silhouette",
        "Feature value": best["Sil"],
        "Normalized weight": weights[3],
        "Weighted contribution": best["wSil*Sil"],
    },
])

st.dataframe(
    comp_df,
    use_container_width=True,
    hide_index=True
)

st.write(
    f"Final score = "
    f"{best['wA*A']:.4f} + {best['wC*C']:.4f} + "
    f"{best['wE*E']:.4f} + {best['wSil*Sil']:.4f} "
    f"= **{best['score']:.4f}**"
)


# =========================================================
# Candidate table
# =========================================================

st.subheader("All candidate masks under current parameters")

st.dataframe(
    df.drop(columns=[]),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# Optional candidate inspection
# =========================================================

st.subheader("Inspect a specific candidate")

candidate_ids = [int(row["id"]) for row in rows]
selected_id = st.selectbox(
    "Candidate id",
    options=candidate_ids,
    index=candidate_ids.index(int(best["id"])) if int(best["id"]) in candidate_ids else 0
)

selected_row = next(row for row in rows if int(row["id"]) == selected_id)
selected_mask = selected_row["mask"]

inspect_col1, inspect_col2 = st.columns(2)

with inspect_col1:
    st.markdown("**Selected candidate mask**")
    if show_overlay:
        st.image(
            overlay_mask_on_image(raw_image, selected_mask, color=(255, 255, 0), alpha=0.45),
            use_container_width=True
        )
    else:
        st.image(mask_to_pil(selected_mask), use_container_width=True)

with inspect_col2:
    st.markdown("**Candidate score details**")
    detail_df = pd.DataFrame([
        {
            "id": selected_row["id"],
            "box_id": selected_row["box_id"],
            "rank_in_box": selected_row["rank_in_box"],
            "sam_score": selected_row["sam_score"],
            "score": selected_row["score"],
            "IoU_vs_GT": selected_row["IoU_vs_GT"],
            "A": selected_row["A"],
            "C": selected_row["C"],
            "E": selected_row["E"],
            "Sil": selected_row["Sil"],
            "wA*A": selected_row["wA*A"],
            "wC*C": selected_row["wC*C"],
            "wE*E": selected_row["wE*E"],
            "wSil*Sil": selected_row["wSil*Sil"],
        }
    ])

    st.dataframe(
        detail_df,
        use_container_width=True,
        hide_index=True
    )