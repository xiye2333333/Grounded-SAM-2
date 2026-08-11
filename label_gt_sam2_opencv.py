import os
import json
from pathlib import Path
import numpy as np
import cv2
import torch
import pycocotools.mask as mask_util

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def coco_rle_from_bool(mask_bool: np.ndarray):
    rle = mask_util.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def bool_from_png(png_path: Path):
    m = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 127)


def overlay_mask_bgr(img_bgr: np.ndarray, mask_bool: np.ndarray, alpha=0.45):
    # green overlay
    overlay = img_bgr.copy()
    overlay[mask_bool] = (0, 255, 0)
    out = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    return out


class MaskEditor:
    """
    OpenCV interactive editor:
    - Left mouse: add (paint foreground)
    - Right mouse: erase (paint background)
    Keys:
      b: select ROI box + run SAM2 init
      r: reset mask (clear)
      ] / [: brush size +/-
      s: save GT
      n: next image
      p: prev image
      q/ESC: quit
    """
    def __init__(self, predictor: SAM2ImagePredictor, device: str):
        self.predictor = predictor
        self.device = device
        self.brush = 18
        self.drawing = False
        self.mode = "add"  # or "erase"
        self.mask_bool = None
        self.box_xyxy = None
        self.img_bgr = None
        self.img_rgb = None
        self.H = None
        self.W = None
        self.win = "GT Labeler (SAM2 + Paint)"

    def set_image(self, img_bgr: np.ndarray):
        self.img_bgr = img_bgr
        self.img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.H, self.W = img_bgr.shape[:2]
        self.mask_bool = None
        self.box_xyxy = None

    def load_mask(self, mask_bool: np.ndarray, box_xyxy=None):
        self.mask_bool = mask_bool.copy().astype(bool)
        self.box_xyxy = box_xyxy

    def _ensure_mask(self):
        if self.mask_bool is None:
            self.mask_bool = np.zeros((self.H, self.W), dtype=bool)

    def _paint_circle(self, x, y, add=True):
        self._ensure_mask()
        rr = self.brush
        yy, xx = np.ogrid[:self.H, :self.W]
        circle = (xx - x) ** 2 + (yy - y) ** 2 <= rr ** 2
        if add:
            self.mask_bool[circle] = True
        else:
            self.mask_bool[circle] = False

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.mode = "add"
            self._paint_circle(x, y, add=True)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.mode = "erase"
            self._paint_circle(x, y, add=False)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            if self.mode == "add":
                self._paint_circle(x, y, add=True)
            else:
                self._paint_circle(x, y, add=False)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.drawing = False

    def select_roi_and_run_sam2(self):
        # ROI selection (returns x,y,w,h)
        roi = cv2.selectROI(self.win, self.img_bgr, fromCenter=False, showCrosshair=True)
        # ✅ IMPORTANT: selectROI may reset mouse callback on some platforms (Windows)
        cv2.setMouseCallback(self.win, self.mouse_cb)
        x, y, w, h = roi
        if w <= 1 or h <= 1:
            print("ROI cancelled.")
            return False
        x1, y1 = float(np.clip(x, 0, self.W - 1)), float(np.clip(y, 0, self.H - 1))
        x2, y2 = float(np.clip(x + w, 0, self.W - 1)), float(np.clip(y + h, 0, self.H - 1))
        self.box_xyxy = [x1, y1, x2, y2]

        # Run SAM2
        self.predictor.set_image(self.img_rgb)
        box = np.array(self.box_xyxy, dtype=np.float32)[None, :]

        with torch.autocast(
            device_type=self.device,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        ):
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=True,
            )

        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
        scores = scores.reshape(-1)
        best = int(np.argmax(scores))
        self.mask_bool = masks[best].astype(bool)
        print(f"SAM2 init done. best_idx={best}, score={float(scores[best]):.4f}")
        return True

    def render(self):
        base = self.img_bgr.copy()
        if self.mask_bool is not None:
            base = overlay_mask_bgr(base, self.mask_bool, alpha=0.45)

        # draw box if exists
        if self.box_xyxy is not None:
            x1, y1, x2, y2 = map(int, self.box_xyxy)
            cv2.rectangle(base, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # HUD text
        fg_px = int(self.mask_bool.sum()) if self.mask_bool is not None else 0
        txt = f"brush={self.brush} | fg_px={fg_px} | keys: b ROI+SAM2, L add, R erase, s save, n/p, q"
        cv2.putText(base, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 3, cv2.LINE_AA)
        cv2.putText(base, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
        return base


def main():
    # -------------------------
    # User configs (edit here)
    # -------------------------
    IMG_DIR = Path(r"D:\uwb thesis\code\Grounded-SAM-2\assets\images")
    GT_DIR = Path(r"D:\uwb thesis\code\Grounded-SAM-2\GT_masks")
    GT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_FP = GT_DIR / "_progress.json"

    SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    PROMPT_TEXT = "seat"  # stored in json metadata only

    # -------------------------
    # Load file list
    # -------------------------
    images_all = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
    if not images_all:
        raise RuntimeError(f"No images found in {IMG_DIR}")

    def gt_png_path(img_path: Path) -> Path:
        return GT_DIR / f"{img_path.stem}_gt.png"

    def gt_json_path(img_path: Path) -> Path:
        return GT_DIR / f"{img_path.stem}_gt.json"

    def is_labeled(img_path: Path) -> bool:
        return gt_png_path(img_path).exists()

    # load progress
    start_idx = 0
    if PROGRESS_FP.exists():
        try:
            prog = json.loads(PROGRESS_FP.read_text(encoding="utf-8"))
            if isinstance(prog.get("last_index"), int):
                start_idx = int(np.clip(prog["last_index"], 0, len(images_all) - 1))
        except Exception:
            pass

    # -------------------------
    # Load SAM2
    # -------------------------
    torch.set_float32_matmul_precision("high")
    print("Loading SAM2...")
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    editor = MaskEditor(predictor, DEVICE)

    cv2.namedWindow(editor.win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(editor.win, editor.mouse_cb)

    idx = start_idx

    while 0 <= idx < len(images_all):
        img_path = images_all[idx]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Failed to read: {img_path}")
            idx += 1
            continue

        editor.set_image(img_bgr)

        # if already labeled, preload existing mask
        out_png = gt_png_path(img_path)
        out_json = gt_json_path(img_path)
        if out_png.exists():
            m = bool_from_png(out_png)
            if m is not None:
                # try read box from json
                box = None
                if out_json.exists():
                    try:
                        meta = json.loads(out_json.read_text(encoding="utf-8"))
                        box = meta.get("box_xyxy", None)
                    except Exception:
                        pass
                editor.load_mask(m, box_xyxy=box)
                print(f"[{idx+1}/{len(images_all)}] Loaded existing GT: {out_png.name}")
        else:
            print(f"[{idx+1}/{len(images_all)}] Unlabeled: {img_path.name} (press 'b' to init SAM2)")

        while True:
            vis = editor.render()
            cv2.imshow(editor.win, vis)
            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord('q')):  # ESC / q
                # save progress and quit
                PROGRESS_FP.write_text(json.dumps({
                    "last_index": idx,
                    "last_image": img_path.name
                }, indent=2), encoding="utf-8")
                cv2.destroyAllWindows()
                print("Quit. Progress saved.")
                return

            elif key == ord(']'):
                editor.brush = int(min(editor.brush + 2, 200))
            elif key == ord('['):
                editor.brush = int(max(editor.brush - 2, 1))

            elif key == ord('r'):
                editor.mask_bool = np.zeros((editor.H, editor.W), dtype=bool)
                editor.box_xyxy = None
                print("Mask reset.")

            elif key == ord('b'):
                editor.select_roi_and_run_sam2()

            elif key == ord('s'):
                if editor.mask_bool is None:
                    print("No mask to save.")
                    continue

                # save png (0/255)
                cv2.imwrite(str(out_png), (editor.mask_bool.astype(np.uint8) * 255))

                # save json metadata
                rle = coco_rle_from_bool(editor.mask_bool)
                payload = {
                    "image_path": str(img_path),
                    "gt_mask_png": str(out_png),
                    "width": int(editor.W),
                    "height": int(editor.H),
                    "prompt_text": PROMPT_TEXT,
                    "box_xyxy": editor.box_xyxy,
                    "mask_fg_pixels": int(editor.mask_bool.sum()),
                    "mask_fg_ratio": float(editor.mask_bool.mean()),
                    "segmentation_rle": rle
                }
                out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(f"Saved GT: {out_png.name} + {out_json.name}")

            elif key == ord('n'):
                # next (prefer next unlabeled if possible)
                PROGRESS_FP.write_text(json.dumps({
                    "last_index": min(idx + 1, len(images_all) - 1),
                    "last_image": img_path.name
                }, indent=2), encoding="utf-8")
                idx += 1
                break

            elif key == ord('p'):
                PROGRESS_FP.write_text(json.dumps({
                    "last_index": max(idx - 1, 0),
                    "last_image": img_path.name
                }, indent=2), encoding="utf-8")
                idx -= 1
                break

            elif key == ord('u'):
                # jump to next unlabeled
                j = idx + 1
                while j < len(images_all) and is_labeled(images_all[j]):
                    j += 1
                if j < len(images_all):
                    idx = j
                    break
                else:
                    print("No more unlabeled images ahead.")

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
