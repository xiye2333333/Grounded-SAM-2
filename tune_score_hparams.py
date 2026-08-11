"""
Optuna 版本的离线超参调优（多类别）：
- 使用预计算好的 precomputed_masks/class_{class_id}/*.npz
- 使用 Optuna (TPE) 做贝叶斯优化，最大化 多个 class 上总体 ours_mean_iou
- 每个 trial 的参数与结果写入 CSV
- 在最优参数下，进行一遍 Wilcoxon signed-rank 检验，并写入单独 CSV

用法示例：
    python tune_score_hparams.py --class-ids 7 8 12 --trials 50
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import optuna
from scipy.stats import wilcoxon  # ⭐ 新增：Wilcoxon 检验

import eval_SAM_with_Score as bench  # 你的 benchmark 模块（有 compute_scores_new, compute_iou）


# 和 objective 中保持一致的常数（方便复用）
GLOBAL_MIN_AREA_RATIO = 0.15
GLOBAL_Q_BORDER = 0.15


# ===========================
# 数据结构：仅用来打印方便
# ===========================

@dataclass
class ScoreParams:
    min_area_ratio: float
    k_area: float
    t_area: float
    w_area: float
    w_center: float
    w_border: float
    w_sil: float
    sigmoid_on: bool


# ===========================
# 预加载 npz 数据（避免反复读盘）
# ===========================

def load_precomputed_class_data(precomputed_root: Path, class_id: int):
    """
    从 precomputed_root/class_{class_id} 中读取所有 npz，
    返回一个 list，每个元素是 dict：
        {
            "class_id": int,
            "image_id": str,
            "gt_mask": (H, W) bool,
            "masks":   (K, H, W) bool,
            "sam_confs": (K,) float
        }
    """
    class_dir = precomputed_root / f"class_{class_id}"
    npz_files = sorted(class_dir.glob("*.npz"))

    if not npz_files:
        raise RuntimeError(f"No npz files found in {class_dir}. 请先跑 precompute_sam_masks.py")

    data_list = []

    for npz_path in npz_files:
        d = np.load(npz_path, allow_pickle=True)
        gt_mask = d["gt_mask"].astype(bool)
        masks = d["masks"].astype(bool)
        sam_confs = d["sam_confs"].astype(float)

        if masks.ndim != 3 or masks.shape[0] == 0:
            continue

        img_id = d.get("image_id", npz_path.stem)
        if isinstance(img_id, np.ndarray):
            img_id = img_id.item()
        img_id = str(img_id)

        data_list.append(
            {
                "class_id": class_id,
                "image_id": img_id,
                "gt_mask": gt_mask,
                "masks": masks,
                "sam_confs": sam_confs,
            }
        )

    if not data_list:
        raise RuntimeError(f"All npz in {class_dir} had empty masks. 请检查预计算结果。")

    print(f"[Class {class_id}] Loaded {len(data_list)} images from {class_dir}")
    return data_list


def load_precomputed_multi(precomputed_root: Path, class_ids):
    """
    一次性把多个 class 的数据都读进来，返回一个大 list。
    """
    all_data = []
    for cid in class_ids:
        data_list = load_precomputed_class_data(precomputed_root, cid)
        all_data.extend(data_list)

    print(f"[Multi-class] Total images loaded: {len(all_data)}")
    return all_data


# ===========================
#  Optuna objective（多类别）
# ===========================

def make_objective(class_ids, precomputed_root: Path):
    """
    返回给 Optuna 用的 objective(trial) 函数。
    - 支持多个 class，一套参数在所有 class 上共享；
    - 目标是所有图像上的总体 ours_mean_iou。
    """

    # 预加载所有类别的数据，只做一次
    data_list = load_precomputed_multi(precomputed_root, class_ids)

    def objective(trial: optuna.Trial) -> float:
        # 1. 让 Optuna 决定一组参数

        # min_area_ratio 固定为某个值（当前语义：GT 占比低于该值的样本整张跳过）
        # 如果以后想重新开启搜索，可：
        # min_area_ratio = trial.suggest_float("min_area_ratio", 0.0, 0.3)
        min_area_ratio = GLOBAL_MIN_AREA_RATIO

        # sigmoid 参数
        k_area = trial.suggest_float("k_area", 0.0, 15.0)
        t_area = trial.suggest_float("t_area", 0.0, 1.0)

        # 权重：0~1，不归一化（按你现在的设计）
        w_area = trial.suggest_float("W_AREA",   0.0, 1.0)
        w_center = trial.suggest_float("W_CENTER", 0.0, 1.0)
        w_border = trial.suggest_float("W_BORDER", 0.0, 1.0)
        w_sil = trial.suggest_float("W_SIL",    0.0, 1.0)

        # sigmoid_on 固定为 True（你现在的设定）
        # sigmoid_on = trial.suggest_categorical("sigmoid_on", [True, False])
        sigmoid_on = True

        # 写回到 bench 的全局变量中
        bench.MIN_AREA_RATIO = min_area_ratio
        bench.k_area = k_area
        bench.t_area = t_area
        bench.W_AREA = w_area
        bench.W_CENTER = w_center
        bench.W_BORDER = w_border
        bench.W_SIL = w_sil
        bench.sigmoid_isOn = sigmoid_on

        # 2. 对所有预计算图像，离线计算 mean IoU（整体 + 分 class）

        ious_sam = []
        ious_ours = []

        # 按 class 单独记录，方便事后分析
        per_class_sam = {cid: [] for cid in class_ids}
        per_class_ours = {cid: [] for cid in class_ids}

        for item in data_list:
            cid = item["class_id"]
            gt = item["gt_mask"]          # (H, W) bool
            masks = item["masks"]         # (K, H, W) bool
            sam_confs = item["sam_confs"] # (K,)

            # === skip small GT masks（GT 占比小于 min_area_ratio 的整张跳过）===
            gt_area_ratio = gt.sum() / (gt.size)
            if gt_area_ratio < min_area_ratio:
                continue

            K, H, W = masks.shape

            # --- SAM baseline: 用 sam_conf 选 Top1 ---
            sam_best_idx = int(np.argmax(sam_confs))
            pred_sam = masks[sam_best_idx]
            iou_sam = bench.compute_iou(pred_sam, gt)

            # --- 我们的 Score: 用当前参数的 score 选 Top1 ---
            masks_list = [masks[i] for i in range(K)]
            scores, _ = bench.compute_scores_new(
                masks_list,
                W=W,
                H=H,
                q_border=GLOBAL_Q_BORDER,  # 这里先固定，也可以之后再加成 hyperparam
                sigmoid_isOn=bench.sigmoid_isOn,
            )
            scores = np.array(scores)
            ours_best_idx = int(np.argmax(scores))
            pred_ours = masks[ours_best_idx]
            iou_ours = bench.compute_iou(pred_ours, gt)

            ious_sam.append(iou_sam)
            ious_ours.append(iou_ours)

            per_class_sam[cid].append(iou_sam)
            per_class_ours[cid].append(iou_ours)

        if not ious_ours:
            # 如果所有样本都被跳过，返回 0（Optuna 会认为很差）
            sam_mean_iou = 0.0
            ours_mean_iou = 0.0
        else:
            sam_mean_iou = float(np.mean(ious_sam))
            ours_mean_iou = float(np.mean(ious_ours))

        # 每个 class 的 mean IoU
        per_class_sam_mean = {
            cid: (float(np.mean(v)) if v else 0.0) for cid, v in per_class_sam.items()
        }
        per_class_ours_mean = {
            cid: (float(np.mean(v)) if v else 0.0) for cid, v in per_class_ours.items()
        }

        # 把这些信息存进 trial.user_attrs，方便之后导出到 CSV
        trial.set_user_attr("sam_mean_iou", sam_mean_iou)
        trial.set_user_attr("ours_mean_iou", ours_mean_iou)
        trial.set_user_attr("num_images", len(ious_ours))  # 实际参与统计的样本数
        trial.set_user_attr("class_ids", class_ids)
        trial.set_user_attr("per_class_sam_mean_iou", per_class_sam_mean)
        trial.set_user_attr("per_class_ours_mean_iou", per_class_ours_mean)

        print(
            f"[Trial {trial.number}] classes {class_ids}, "
            f"num_images={len(ious_ours)}, "
            f"SAM mIoU={sam_mean_iou:.4f}, "
            f"Ours mIoU={ours_mean_iou:.4f}"
        )
        for cid in class_ids:
            print(
                f"   - class {cid}: SAM={per_class_sam_mean[cid]:.4f}, "
                f"Ours={per_class_ours_mean[cid]:.4f}"
            )

        # 目标：最大化总体 ours_mean_iou
        return ours_mean_iou

    return objective


# ===========================
#  导出 CSV（Optuna trial 记录）
# ===========================

def export_study_to_csv(study: optuna.Study, class_ids, csv_path: Path):
    """
    把所有 trial 的参数与结果导出到 CSV。
    per-class IoU 用 JSON 字符串保存。
    """

    fieldnames = [
        "trial",
        "class_ids",
        "num_images",
        "min_area_ratio",
        "k_area",
        "t_area",
        "W_AREA",
        "W_CENTER",
        "W_BORDER",
        "W_SIL",
        "sigmoid_on",
        "sam_mean_iou",
        "ours_mean_iou",
        "per_class_sam_mean_iou",
        "per_class_ours_mean_iou",
        "objective_value",
        "state",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for t in study.trials:
            # 通用取法：不管 COMPLETE / PRUNED，都用 get，防止 KeyError
            params = t.params
            ua = t.user_attrs

            row = {
                "trial": t.number,
                "class_ids": ",".join(str(cid) for cid in ua.get("class_ids", class_ids)),
                "num_images": ua.get("num_images", 0),
                "min_area_ratio": params.get("min_area_ratio", GLOBAL_MIN_AREA_RATIO),
                "k_area": params.get("k_area", None),
                "t_area": params.get("t_area", None),
                "W_AREA": params.get("W_AREA", None),
                "W_CENTER": params.get("W_CENTER", None),
                "W_BORDER": params.get("W_BORDER", None),
                "W_SIL": params.get("W_SIL", None),
                "sigmoid_on": params.get("sigmoid_on", True),
                "sam_mean_iou": ua.get("sam_mean_iou", None),
                "ours_mean_iou": ua.get("ours_mean_iou", None),
                "per_class_sam_mean_iou": json.dumps(ua.get("per_class_sam_mean_iou", {})),
                "per_class_ours_mean_iou": json.dumps(ua.get("per_class_ours_mean_iou", {})),
                "objective_value": t.value,
                "state": str(t.state),
            }
            writer.writerow(row)

    print(f"CSV exported to {csv_path}")


# ===========================
#  Wilcoxon: 用最优参数重跑一遍
# ===========================

def compute_wilcoxon_for_best(class_ids, precomputed_root: Path, best_params: dict):
    """
    使用 Optuna 找到的 best_params：
    - 设置 scoring 参数
    - 在所有指定 class 的数据上重新计算 iou_sam / iou_ours
    - 做 Wilcoxon signed-rank test (one-sided, ours > sam)
    返回一个 dict，包含 p-value 等信息。
    """

    min_area_ratio = GLOBAL_MIN_AREA_RATIO
    sigmoid_on = best_params.get("sigmoid_on", True)

    # 写回到 bench 的全局变量中（确保和调参时一致）
    bench.MIN_AREA_RATIO = min_area_ratio
    bench.k_area = best_params["k_area"]
    bench.t_area = best_params["t_area"]
    bench.W_AREA = best_params["W_AREA"]
    bench.W_CENTER = best_params["W_CENTER"]
    bench.W_BORDER = best_params["W_BORDER"]
    bench.W_SIL = best_params["W_SIL"]
    bench.sigmoid_isOn = sigmoid_on

    data_list = load_precomputed_multi(precomputed_root, class_ids)

    ious_sam = []
    ious_ours = []

    for item in data_list:
        gt = item["gt_mask"]
        masks = item["masks"]
        sam_confs = item["sam_confs"]

        # GT 太小的样本跳过
        gt_area_ratio = gt.sum() / gt.size
        if gt_area_ratio < min_area_ratio:
            continue

        K, H, W = masks.shape

        # SAM：sam_conf top1
        sam_best_idx = int(np.argmax(sam_confs))
        pred_sam = masks[sam_best_idx]
        iou_sam = bench.compute_iou(pred_sam, gt)

        # Ours：score top1（在 best_params 下）
        masks_list = [masks[i] for i in range(K)]
        scores, _ = bench.compute_scores_new(
            masks_list,
            W=W,
            H=H,
            q_border=GLOBAL_Q_BORDER,
            sigmoid_isOn=bench.sigmoid_isOn,
        )
        scores = np.array(scores)
        ours_best_idx = int(np.argmax(scores))
        pred_ours = masks[ours_best_idx]
        iou_ours = bench.compute_iou(pred_ours, gt)

        ious_sam.append(iou_sam)
        ious_ours.append(iou_ours)

    ious_sam = np.array(ious_sam, dtype=float)
    ious_ours = np.array(ious_ours, dtype=float)

    num_all = len(ious_sam)

    if num_all == 0:
        # 没有有效样本
        return {
            "num_all": 0,
            "num_effective": 0,
            "sam_mean": 0.0,
            "ours_mean": 0.0,
            "stat": 0.0,
            "p_value": 1.0,
            "alternative": "greater",
        }

    # 只保留 diff != 0 的样本（Wilcoxon 标准做法）
    diffs = ious_ours - ious_sam
    mask = diffs != 0
    ious_sam_eff = ious_sam[mask]
    ious_ours_eff = ious_ours[mask]
    num_eff = len(ious_sam_eff)

    if num_eff == 0:
        # 完全一模一样
        return {
            "num_all": num_all,
            "num_effective": 0,
            "sam_mean": float(np.mean(ious_sam)),
            "ours_mean": float(np.mean(ious_ours)),
            "stat": 0.0,
            "p_value": 1.0,
            "alternative": "greater",
        }

    # Wilcoxon signed-rank test（单边：ours > sam）
    stat, p_value = wilcoxon(
        ious_ours_eff,
        ious_sam_eff,
        alternative="greater",
    )

    return {
        "num_all": num_all,
        "num_effective": num_eff,
        "sam_mean": float(np.mean(ious_sam)),
        "ours_mean": float(np.mean(ious_ours)),
        "stat": float(stat),
        "p_value": float(p_value),
        "alternative": "greater",
    }


def export_wilcoxon_summary(class_ids, csv_base_path: Path, study: optuna.Study, summary: dict):
    """
    把 Wilcoxon 的结果写入一个简短的 CSV：
    默认文件名：原 trial CSV 加 _wilcoxon 后缀。
    """
    if csv_base_path.suffix:
        wilcoxon_csv = csv_base_path.with_name(csv_base_path.stem + "_wilcoxon.csv")
    else:
        wilcoxon_csv = csv_base_path.with_name(str(csv_base_path) + "_wilcoxon.csv")

    fieldnames = [
        "class_ids",
        "min_area_ratio",
        "q_border",
        "num_all_samples",
        "num_effective_pairs",
        "sam_mean_iou",
        "ours_mean_iou",
        "wilcoxon_stat",
        "p_value",
        "alternative",
        "n_trials",
        "best_params_json",
    ]

    with open(wilcoxon_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        row = {
            "class_ids": ",".join(str(cid) for cid in class_ids),
            "min_area_ratio": GLOBAL_MIN_AREA_RATIO,
            "q_border": GLOBAL_Q_BORDER,
            "num_all_samples": summary["num_all"],
            "num_effective_pairs": summary["num_effective"],
            "sam_mean_iou": summary["sam_mean"],
            "ours_mean_iou": summary["ours_mean"],
            "wilcoxon_stat": summary["stat"],
            "p_value": summary["p_value"],
            "alternative": summary["alternative"],
            "n_trials": len(study.trials),
            "best_params_json": json.dumps(study.best_params),
        }
        writer.writerow(row)

    print(f"Wilcoxon summary CSV exported to {wilcoxon_csv}")
    print(
        f"🔬 Wilcoxon (ours > sam): p = {summary['p_value']:.6g}, "
        f"W = {summary['stat']:.3f}, "
        f"n_eff = {summary['num_effective']}, "
        f"mean_iou: SAM={summary['sam_mean']:.4f}, Ours={summary['ours_mean']:.4f}"
    )


# ===========================
#  main & CLI
# ===========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna-based offline hyperparameter tuning for the scoring function, using precomputed SAM masks (.npz)."
    )
    parser.add_argument(
        "--class-ids", type=int, nargs="+", required=True,
        help="VOC class ids to tune for (e.g., 7 8 12)."
    )
    parser.add_argument(
        "--precomputed-root", type=str, default="precomputed_masks",
        help="Root directory of precomputed npz files (default: precomputed_masks).",
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Number of Optuna trials (default: 50).",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Output CSV path (default: tuning_optuna_classes_{ids}.csv).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    class_ids = args.class_ids
    precomputed_root = Path(args.precomputed_root)

    if args.csv is not None:
        csv_path = Path(args.csv)
    else:
        ids_str = "_".join(str(cid) for cid in class_ids)
        csv_path = Path(f"tuning_optuna_classes_{ids_str}.csv")

    objective = make_objective(class_ids, precomputed_root)

    # 创建 study：最大化 ours_mean_iou
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    print("\n==============================")
    print(f"[Classes {class_ids}] Optuna tuning finished.")
    print(f"Best value (ours_mean_iou): {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 导出所有 trial 的 CSV
    export_study_to_csv(study, class_ids, csv_path)

    # ⭐ 用最优参数做一次 Wilcoxon，写入单独 CSV
    summary = compute_wilcoxon_for_best(class_ids, precomputed_root, study.best_params)
    export_wilcoxon_summary(class_ids, csv_path, study, summary)


if __name__ == "__main__":
    main()
