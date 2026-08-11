
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Basic utilities
# =========================================================
def safe_read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def list_sequence_dirs(out_root: Path) -> List[Path]:
    """
    Sequence folders are identified by the existence of /json
    """
    seq_dirs = []
    for p in sorted(out_root.iterdir()):
        if p.is_dir() and (p / "json").exists():
            seq_dirs.append(p)
    return seq_dirs


def save_bar_chart(x_labels, values, title, ylabel, out_path: Path, rotation=90):
    plt.figure(figsize=(12, 5))
    plt.bar(x_labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_grouped_bar_chart(x_labels, series_dict, title, ylabel, out_path: Path, rotation=90):
    plt.figure(figsize=(13, 6))
    x = np.arange(len(x_labels))
    n = len(series_dict)
    width = 0.8 / max(n, 1)

    for i, (name, values) in enumerate(series_dict.items()):
        plt.bar(x + i * width - (n - 1) * width / 2, values, width=width, label=name)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x, x_labels, rotation=rotation, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_scatter(x, y, title, xlabel, ylabel, out_path: Path, labels=None):
    plt.figure(figsize=(7, 6))
    plt.scatter(x, y)
    if labels is not None:
        for xi, yi, lb in zip(x, y, labels):
            plt.annotate(str(lb), (xi, yi), fontsize=7, alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_histogram(values, title, xlabel, ylabel, out_path: Path, bins=40):
    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# =========================================================
# Data loading / flattening
# =========================================================
def load_sequence_artifacts(seq_dir: Path) -> Dict[str, Optional[Any]]:
    return {
        "optimized_summary": safe_read_json(seq_dir / "eval_summary_optimized.json"),
        "default_summary": safe_read_json(seq_dir / "eval_summary_default_no_gt.json"),
        "ablation_summary": safe_read_json(seq_dir / "ablation_summary_default_no_gt.json"),
        "optimized_params": safe_read_json(seq_dir / "optimized_params.json"),
    }


def flatten_main_rows(seq_name: str,
                      default_sum: Optional[dict],
                      optimized_sum: Optional[dict]) -> List[Dict[str, Any]]:
    rows = []

    if default_sum is not None:
        rows.append({
            "sequence": seq_name,
            "setting": "default_no_gt",
            "baseline_mean_iou": default_sum["baseline_iou"]["mean"],
            "baseline_min_iou": default_sum["baseline_iou"]["min"],
            "baseline_max_iou": default_sum["baseline_iou"]["max"],
            "scored_mean_iou": default_sum["scored_iou"]["mean"],
            "scored_min_iou": default_sum["scored_iou"]["min"],
            "scored_max_iou": default_sum["scored_iou"]["max"],
            "mean_delta_vs_sam": default_sum["scored_iou"]["mean"] - default_sum["baseline_iou"]["mean"],
            "n_frames": default_sum["scored_iou"]["n"],
            "best_improve_frame": default_sum["best_improvement"]["key"],
            "best_improve_delta": default_sum["best_improvement"]["delta"],
            "worst_drop_frame": default_sum["worst_drop"]["key"],
            "worst_drop_delta": default_sum["worst_drop"]["delta"],
        })

    if optimized_sum is not None:
        rows.append({
            "sequence": seq_name,
            "setting": "optimized",
            "baseline_mean_iou": optimized_sum["baseline_iou"]["mean"],
            "baseline_min_iou": optimized_sum["baseline_iou"]["min"],
            "baseline_max_iou": optimized_sum["baseline_iou"]["max"],
            "scored_mean_iou": optimized_sum["scored_iou"]["mean"],
            "scored_min_iou": optimized_sum["scored_iou"]["min"],
            "scored_max_iou": optimized_sum["scored_iou"]["max"],
            "mean_delta_vs_sam": optimized_sum["scored_iou"]["mean"] - optimized_sum["baseline_iou"]["mean"],
            "n_frames": optimized_sum["scored_iou"]["n"],
            "best_improve_frame": optimized_sum["best_improvement"]["key"],
            "best_improve_delta": optimized_sum["best_improvement"]["delta"],
            "worst_drop_frame": optimized_sum["worst_drop"]["key"],
            "worst_drop_delta": optimized_sum["worst_drop"]["delta"],
        })

    return rows


def flatten_ablation_rows(seq_name: str, ablation_sum: Optional[dict]) -> List[Dict[str, Any]]:
    rows = []
    if ablation_sum is None:
        return rows

    for abl_name, item in ablation_sum.get("ablations", {}).items():
        rows.append({
            "sequence": seq_name,
            "ablation": abl_name,
            "baseline_mean_iou": item["baseline_iou"]["mean"],
            "scored_mean_iou": item["scored_iou"]["mean"],
            "mean_delta_vs_sam": item["mean_delta_vs_sam_baseline"],
            "n_frames": item["n_frames_evaluated"],
            "best_improve_frame": item["best_improvement"]["key"],
            "best_improve_delta": item["best_improvement"]["delta"],
            "worst_drop_frame": item["worst_drop"]["key"],
            "worst_drop_delta": item["worst_drop"]["delta"],
            "w_area": item["weights_after_normalize"][0],
            "w_center": item["weights_after_normalize"][1],
            "w_border": item["weights_after_normalize"][2],
            "w_silhouette": item["weights_after_normalize"][3],
            "q_border": item["q_border"],
            "t_area": item["t_area"],
        })
    return rows


def flatten_optimized_params_rows(seq_name: str, opt_params: Optional[dict]) -> Optional[Dict[str, Any]]:
    if opt_params is None:
        return None

    baseline = opt_params.get("baseline_for_optimization", opt_params.get("baseline", {}))
    optimized = opt_params.get("optimized", {})

    return {
        "sequence": seq_name,
        "n_total_frames_with_gt": opt_params.get("n_total_frames_with_gt"),
        "n_used_for_opt": opt_params.get("n_used_for_opt"),
        "gt_ratio": opt_params.get("gt_ratio"),
        "opt_baseline_w_area": baseline.get("w", [None, None, None, None])[0],
        "opt_baseline_w_center": baseline.get("w", [None, None, None, None])[1],
        "opt_baseline_w_border": baseline.get("w", [None, None, None, None])[2],
        "opt_baseline_w_silhouette": baseline.get("w", [None, None, None, None])[3],
        "opt_baseline_q_border": baseline.get("q_border"),
        "opt_baseline_t_area": baseline.get("t_area"),
        "opt_baseline_train_meanIoU": baseline.get("train_meanIoU"),
        "opt_baseline_val_meanIoU": baseline.get("val_meanIoU"),
        "optimized_w_area": optimized.get("w", [None, None, None, None])[0],
        "optimized_w_center": optimized.get("w", [None, None, None, None])[1],
        "optimized_w_border": optimized.get("w", [None, None, None, None])[2],
        "optimized_w_silhouette": optimized.get("w", [None, None, None, None])[3],
        "optimized_q_border": optimized.get("q_border"),
        "optimized_t_area": optimized.get("t_area"),
        "optimized_train_meanIoU": optimized.get("train_meanIoU"),
        "optimized_val_meanIoU": optimized.get("val_meanIoU"),
    }


def collect_per_frame_results(seq_dirs: List[Path], mode: str) -> pd.DataFrame:
    rows = []
    for seq_dir in seq_dirs:
        csv_path = seq_dir / f"eval_per_frame_{mode}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["sequence"] = seq_dir.name
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# =========================================================
# Section 4.1 Experimental Setup
# Here we only save a compact dataset/setup inventory based
# on available result files.
# =========================================================
def section_41_experimental_setup(seq_dirs: List[Path], out_dir: Path):
    rows = []
    for seq_dir in seq_dirs:
        n_result_json = len(list((seq_dir / "json").glob("*_result.json"))) if (seq_dir / "json").exists() else 0
        n_gt_json = len(list((seq_dir / "gt_json").glob("*_gt.json"))) if (seq_dir / "gt_json").exists() else 0
        n_masks_png = len(list((seq_dir / "masks").glob("*.png"))) if (seq_dir / "masks").exists() else 0
        rows.append({
            "sequence": seq_dir.name,
            "n_result_json": n_result_json,
            "n_gt_json": n_gt_json,
            "n_masks_png": n_masks_png,
        })

    df = pd.DataFrame(rows).sort_values("sequence")
    df.to_csv(out_dir / "setup_inventory.csv", index=False)

    summary = {
        "num_sequences": int(len(df)),
        "total_result_json": int(df["n_result_json"].sum()) if not df.empty else 0,
        "total_gt_json": int(df["n_gt_json"].sum()) if not df.empty else 0,
        "total_mask_png": int(df["n_masks_png"].sum()) if not df.empty else 0,
    }
    with open(out_dir / "setup_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return df, summary


# =========================================================
# Section 4.2 Main Evaluation:
# Training-Free Mask Ranking Performance
# =========================================================
def section_42_main_evaluation(df_main: pd.DataFrame, out_dir: Path, overview_dir: Path):
    if df_main.empty:
        return None

    df_main.to_csv(out_dir / "merged_main_results.csv", index=False)

    df_default = df_main[df_main["setting"] == "default_no_gt"].copy()
    df_optimized = df_main[df_main["setting"] == "optimized"].copy()

    rows = []

    if not df_default.empty:
        sam_mean = df_default["baseline_mean_iou"].mean()
        default_mean = df_default["scored_mean_iou"].mean()

        rows.append({
            "Selection Rule": "SAM confidence",
            "Mean IoU": sam_mean,
            "Gain over SAM": 0.0,
            "Notes": "raw SAM score"
        })
        rows.append({
            "Selection Rule": "Ours (default, no GT)",
            "Mean IoU": default_mean,
            "Gain over SAM": default_mean - sam_mean,
            "Notes": "fixed training-free score"
        })

        if not df_optimized.empty:
            optimized_mean = df_optimized["scored_mean_iou"].mean()
            rows.append({
                "Selection Rule": "Ours (optimized with partial GT)",
                "Mean IoU": optimized_mean,
                "Gain over SAM": optimized_mean - sam_mean,
                "Notes": "sequence-level tuned parameters"
            })

    df_table = pd.DataFrame(rows)
    df_table.to_csv(out_dir / "paper_table_main_quantitative.csv", index=False)

    if not df_default.empty:
        df_default_sorted = df_default.sort_values("mean_delta_vs_sam", ascending=False)

        save_grouped_bar_chart(
            x_labels=df_default_sorted["sequence"].tolist(),
            series_dict={
                "SAM baseline": df_default_sorted["baseline_mean_iou"].tolist(),
                "Ours default (no GT)": df_default_sorted["scored_mean_iou"].tolist(),
            },
            title="Section 4.2: Default training-free scoring vs SAM baseline",
            ylabel="Mean IoU",
            out_path=out_dir / "default_vs_sam_by_sequence.png",
        )

        save_bar_chart(
            x_labels=df_default_sorted["sequence"].tolist(),
            values=df_default_sorted["mean_delta_vs_sam"].tolist(),
            title="Section 4.2: Gain over SAM baseline by sequence",
            ylabel="Mean IoU gain",
            out_path=out_dir / "default_gain_over_sam.png",
        )

        save_scatter(
            x=df_default["baseline_mean_iou"].tolist(),
            y=df_default["mean_delta_vs_sam"].tolist(),
            labels=df_default["sequence"].tolist(),
            title="Section 4.2: Gain vs baseline difficulty",
            xlabel="SAM baseline mean IoU",
            ylabel="Mean IoU gain over SAM",
            out_path=out_dir / "default_gain_scatter.png",
        )

        # copy a compact overview table
        df_default[[
            "sequence", "baseline_mean_iou", "scored_mean_iou", "mean_delta_vs_sam"
        ]].rename(columns={
            "sequence": "Sequence",
            "baseline_mean_iou": "SAM",
            "scored_mean_iou": "Default(No GT)",
            "mean_delta_vs_sam": "Gain"
        }).sort_values("Gain", ascending=False).to_csv(
            overview_dir / "overview_default_vs_sam_table.csv", index=False
        )

    return df_table


# =========================================================
# Section 4.3 Frame-Level Improvement Analysis
# =========================================================
def section_43_frame_level_analysis(seq_dirs: List[Path], out_dir: Path, overview_dir: Path, eps: float = 1e-6):
    df_frame = collect_per_frame_results(seq_dirs, mode="default_no_gt")
    if df_frame.empty:
        return None

    df_frame.to_csv(out_dir / "merged_per_frame_default_no_gt.csv", index=False)

    delta = df_frame["delta"].values
    outcome = np.where(delta > eps, "Improved",
               np.where(delta < -eps, "Worsened", "Unchanged"))

    df_frame["outcome"] = outcome

    summary = (
        df_frame.groupby("outcome", as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    summary["percentage"] = summary["count"] / summary["count"].sum()
    summary.to_csv(out_dir / "frame_outcome_summary.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.bar(summary["outcome"], summary["percentage"])
    plt.title("Section 4.3: Frame-level outcome distribution")
    plt.ylabel("Percentage of frames")
    plt.tight_layout()
    plt.savefig(out_dir / "frame_outcome_distribution_default_no_gt.png", dpi=220)
    plt.close()

    save_histogram(
        values=df_frame["delta"].values,
        title="Section 4.3: Distribution of frame-level IoU improvement",
        xlabel="Delta IoU (ours - SAM baseline)",
        ylabel="Number of frames",
        out_path=out_dir / "frame_delta_histogram_default_no_gt.png",
        bins=40,
    )

    # sequence-level improved ratio
    df_seq = (
        df_frame.groupby("sequence", as_index=False)
        .agg(
            mean_delta=("delta", "mean"),
            improved_ratio=("outcome", lambda x: np.mean(np.array(x) == "Improved")),
            worsened_ratio=("outcome", lambda x: np.mean(np.array(x) == "Worsened")),
        )
        .sort_values("mean_delta", ascending=False)
    )
    df_seq.to_csv(out_dir / "frame_outcome_by_sequence.csv", index=False)

    save_grouped_bar_chart(
        x_labels=df_seq["sequence"].tolist(),
        series_dict={
            "Improved ratio": df_seq["improved_ratio"].tolist(),
            "Worsened ratio": df_seq["worsened_ratio"].tolist(),
        },
        title="Section 4.3: Improved vs worsened frame ratio by sequence",
        ylabel="Ratio",
        out_path=out_dir / "frame_outcome_ratio_by_sequence.png",
    )

    summary.to_csv(overview_dir / "overview_frame_outcome_summary.csv", index=False)
    return df_frame, summary, df_seq


# =========================================================
# Section 4.4 Ablation Study
# =========================================================
def section_44_ablation(df_ablation: pd.DataFrame, out_dir: Path, overview_dir: Path):
    if df_ablation.empty:
        return None

    df_ablation.to_csv(out_dir / "merged_ablation_results.csv", index=False)

    df_agg = (
        df_ablation.groupby("ablation", as_index=False)
        .agg(
            Mean_IoU=("scored_mean_iou", "mean"),
            Gain_vs_SAM=("mean_delta_vs_sam", "mean"),
            Median_Gain_vs_SAM=("mean_delta_vs_sam", "median"),
            Num_Sequences=("sequence", "count"),
        )
    )

    full_row = df_agg[df_agg["ablation"] == "default_full"]
    full_mean = full_row["Mean_IoU"].iloc[0] if not full_row.empty else None
    df_agg["Drop_vs_Full"] = (full_mean - df_agg["Mean_IoU"]) if full_mean is not None else np.nan

    abl_order = [x for x in ["default_full", "no_area", "no_center", "no_border", "no_silhouette"]
                 if x in df_agg["ablation"].tolist()]
    if abl_order:
        df_agg["ablation"] = pd.Categorical(df_agg["ablation"], categories=abl_order, ordered=True)
        df_agg = df_agg.sort_values("ablation")

    df_agg.rename(columns={"ablation": "Variant"}).to_csv(
        out_dir / "paper_table_ablation_detailed.csv", index=False
    )

    save_grouped_bar_chart(
        x_labels=df_agg["ablation"].astype(str).tolist(),
        series_dict={
            "Mean IoU": df_agg["Mean_IoU"].tolist(),
            "Gain vs SAM": df_agg["Gain_vs_SAM"].tolist(),
        },
        title="Section 4.4: Ablation aggregate results",
        ylabel="Score",
        out_path=out_dir / "ablation_aggregate_results.png",
        rotation=20,
    )

    save_bar_chart(
        x_labels=df_agg["ablation"].astype(str).tolist(),
        values=df_agg["Drop_vs_Full"].tolist(),
        title="Section 4.4: Performance drop relative to full score",
        ylabel="Drop vs full",
        out_path=out_dir / "ablation_drop_vs_full.png",
        rotation=20,
    )

    # per-sequence heatmap-like csv pivot
    pivot = df_ablation.pivot_table(
        index="sequence", columns="ablation", values="scored_mean_iou", aggfunc="mean"
    )
    pivot.to_csv(out_dir / "ablation_sequence_by_variant_matrix.csv")

    df_agg.rename(columns={"ablation": "Variant"}).to_csv(
        overview_dir / "overview_ablation_table.csv", index=False
    )
    return df_agg, pivot


# =========================================================
# Section 4.5 Optional Few-Shot / Partial-GT Calibration
# Here we only compare:
#   Default(No GT) vs Optimized(Partial GT)
# No multi-ratio curve is included for now.
# =========================================================
def section_45_partial_gt_calibration(df_main: pd.DataFrame, df_opt_params: pd.DataFrame,
                                      out_dir: Path, overview_dir: Path):
    if df_main.empty:
        return None

    df_default = df_main[df_main["setting"] == "default_no_gt"].copy()
    df_optimized = df_main[df_main["setting"] == "optimized"].copy()

    if df_default.empty or df_optimized.empty:
        return None

    df_compare = df_default.merge(
        df_optimized,
        on="sequence",
        suffixes=("_default", "_optimized")
    )

    df_compare["optimized_gain_over_default"] = (
        df_compare["scored_mean_iou_optimized"] - df_compare["scored_mean_iou_default"]
    )

    keep_cols = [
        "sequence",
        "scored_mean_iou_default",
        "scored_mean_iou_optimized",
        "optimized_gain_over_default"
    ]
    df_paper = df_compare[keep_cols].copy()
    df_paper.columns = ["Sequence", "Default(No GT)", "Optimized(Partial GT)", "Gain"]
    df_paper = df_paper.sort_values("Gain", ascending=False)
    df_paper.to_csv(out_dir / "paper_table_default_vs_optimized.csv", index=False)

    save_grouped_bar_chart(
        x_labels=df_paper["Sequence"].tolist(),
        series_dict={
            "Default(No GT)": df_paper["Default(No GT)"].tolist(),
            "Optimized(Partial GT)": df_paper["Optimized(Partial GT)"].tolist(),
        },
        title="Section 4.5: Default vs partial-GT optimized performance",
        ylabel="Mean IoU",
        out_path=out_dir / "default_vs_optimized_by_sequence.png",
    )

    save_bar_chart(
        x_labels=df_paper["Sequence"].tolist(),
        values=df_paper["Gain"].tolist(),
        title="Section 4.4: Gain from partial-GT optimization",
        ylabel="Mean IoU gain",
        out_path=out_dir / "optimized_gain_over_default.png",
    )

    if not df_opt_params.empty:
        df_opt_params.to_csv(out_dir / "optimized_parameter_records.csv", index=False)

    df_paper.to_csv(overview_dir / "overview_default_vs_optimized_table.csv", index=False)
    return df_paper


# =========================================================
# Overview / summary report
# =========================================================
def write_summary_report(out_dir: Path,
                         setup_summary: dict,
                         df_main: pd.DataFrame,
                         df_ablation: pd.DataFrame):
    lines = []
    lines.append("DAVIS experiment analysis report\n")
    lines.append("=" * 40 + "\n\n")

    lines.append("[Section 4.1] Experimental Setup\n")
    lines.append(f"- Number of sequences: {setup_summary.get('num_sequences', 0)}\n")
    lines.append(f"- Total result json files: {setup_summary.get('total_result_json', 0)}\n")
    lines.append(f"- Total gt json files: {setup_summary.get('total_gt_json', 0)}\n")
    lines.append(f"- Total mask png files: {setup_summary.get('total_mask_png', 0)}\n\n")

    df_default = df_main[df_main["setting"] == "default_no_gt"].copy() if not df_main.empty else pd.DataFrame()
    df_optimized = df_main[df_main["setting"] == "optimized"].copy() if not df_main.empty else pd.DataFrame()

    lines.append("[Section 4.2] Main Evaluation\n")
    if not df_default.empty:
        lines.append(f"- Mean SAM baseline IoU: {df_default['baseline_mean_iou'].mean():.4f}\n")
        lines.append(f"- Mean default(no GT) IoU: {df_default['scored_mean_iou'].mean():.4f}\n")
        lines.append(f"- Mean gain over SAM: {df_default['mean_delta_vs_sam'].mean():.4f}\n")
    if not df_optimized.empty:
        lines.append(f"- Mean optimized(partial GT) IoU: {df_optimized['scored_mean_iou'].mean():.4f}\n")
    lines.append("\n")

    lines.append("[Section 4.4] Ablation\n")
    if not df_ablation.empty:
        abl_rank = df_ablation.groupby("ablation")["scored_mean_iou"].mean().sort_values(ascending=False)
        for name, val in abl_rank.items():
            lines.append(f"- {name}: {val:.4f}\n")
    else:
        lines.append("- No ablation results found.\n")
    lines.append("\n")

    if not df_default.empty and not df_optimized.empty:
        df_cmp = df_default.merge(df_optimized, on="sequence", suffixes=("_default", "_optimized"))
        gain = (df_cmp["scored_mean_iou_optimized"] - df_cmp["scored_mean_iou_default"]).mean()
        lines.append("[Section 4.5] Partial-GT Calibration\n")
        lines.append(f"- Mean gain of optimized over default: {gain:.4f}\n\n")

    (out_dir / "analysis_report.txt").write_text("".join(lines), encoding="utf-8")


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True,
                    help="Root folder of experiment outputs, e.g. D:/.../SAM_results")
    ap.add_argument("--analysis_out", type=str, default="analysis_outputs_paper",
                    help="Folder to save section-organized analysis outputs")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    analysis_out = Path(args.analysis_out)

    # Outer structure:
    # analysis_out/
    #   overview/
    #   section_4_1_setup/
    #   section_4_2_main_eval/
    #   section_4_3_frame_level/
    #   section_4_4_ablation/
    #   section_4_5_partial_gt/
    ensure_dir(analysis_out)
    overview_dir = analysis_out / "overview"
    s41_dir = analysis_out / "section_4_1_setup"
    s42_dir = analysis_out / "section_4_2_main_eval"
    s43_dir = analysis_out / "section_4_3_frame_level"
    s44_dir = analysis_out / "section_4_4_ablation"
    s45_dir = analysis_out / "section_4_5_partial_gt"

    for d in [overview_dir, s41_dir, s42_dir, s43_dir, s44_dir, s45_dir]:
        ensure_dir(d)

    seq_dirs = list_sequence_dirs(out_root)
    if not seq_dirs:
        raise RuntimeError(f"No sequence folders found under: {out_root}")

    main_rows = []
    ablation_rows = []
    opt_param_rows = []

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        artifacts = load_sequence_artifacts(seq_dir)

        main_rows.extend(
            flatten_main_rows(seq_name, artifacts["default_summary"], artifacts["optimized_summary"])
        )
        ablation_rows.extend(
            flatten_ablation_rows(seq_name, artifacts["ablation_summary"])
        )
        opt_row = flatten_optimized_params_rows(seq_name, artifacts["optimized_params"])
        if opt_row is not None:
            opt_param_rows.append(opt_row)

    df_main = pd.DataFrame(main_rows)
    df_ablation = pd.DataFrame(ablation_rows)
    df_opt_params = pd.DataFrame(opt_param_rows)

    # Section 4.1
    _, setup_summary = section_41_experimental_setup(seq_dirs, s41_dir)

    # Section 4.2
    section_42_main_evaluation(df_main, s42_dir, overview_dir)

    # Section 4.3
    section_43_frame_level_analysis(seq_dirs, s43_dir, overview_dir)

    # Section 4.4
    section_44_ablation(df_ablation, s44_dir, overview_dir)

    # Section 4.5
    section_45_partial_gt_calibration(df_main, df_opt_params, s45_dir, overview_dir)

    # Summary report
    write_summary_report(analysis_out, setup_summary, df_main, df_ablation)

    print(f"[DONE] Section-organized analysis saved to: {analysis_out}")


if __name__ == "__main__":
    main()

