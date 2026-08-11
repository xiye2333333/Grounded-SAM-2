import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Basic helpers
# =========================================================
def safe_float(x):
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def safe_int(x):
    try:
        if pd.isna(x):
            return 0
        return int(x)
    except Exception:
        return 0


def safe_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, (int, float)) and np.isfinite(x):
        return bool(x)
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def summary_stat(summary_obj, key, stat_name="mean"):
    obj = summary_obj.get(key, {})
    if isinstance(obj, dict):
        return safe_float(obj.get(stat_name, np.nan))
    return np.nan


def get_nested(d, *keys, default=np.nan):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# Selected B-mask score diagnostics
# =========================================================
def empty_score_stats(prefix):
    return {
        f"{prefix}_mean": np.nan,
        f"{prefix}_variance": np.nan,
        f"{prefix}_std": np.nan,
        f"{prefix}_min": np.nan,
        f"{prefix}_min_frame": "",
        f"{prefix}_max": np.nan,
        f"{prefix}_max_frame": "",
        f"{prefix}_range": np.nan,
    }


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def per_frame_term_stats(per_frame_records):
    """
    Compute sequence-level statistics over the B-selected mask in each frame.
    Scope:
      - per-sequence variance/std of selected-mask A/C/E/S terms
      - across-frame min/max of selected-mask A/C/E/S terms
      - across-frame min/max of selected-mask total score_B
    """
    cols = ["score_B", "A_match", "C_match", "E_match", "S_match"]
    out = {}

    if not per_frame_records:
        for col in cols:
            out.update(empty_score_stats(col))
        return out

    df = pd.DataFrame(per_frame_records)
    if "frame" not in df.columns:
        df["frame"] = ""

    for col in cols:
        vals = numeric_series(df, col)
        valid = vals.dropna()
        if valid.empty:
            out.update(empty_score_stats(col))
            continue

        min_idx = vals.idxmin()
        max_idx = vals.idxmax()
        min_val = float(vals.loc[min_idx])
        max_val = float(vals.loc[max_idx])

        out[f"{col}_mean"] = float(valid.mean())
        out[f"{col}_variance"] = float(valid.var(ddof=0))
        out[f"{col}_std"] = float(valid.std(ddof=0))
        out[f"{col}_min"] = min_val
        out[f"{col}_min_frame"] = str(df.loc[min_idx, "frame"])
        out[f"{col}_max"] = max_val
        out[f"{col}_max_frame"] = str(df.loc[max_idx, "frame"])
        out[f"{col}_range"] = float(max_val - min_val)

    return out


# =========================================================
# Candidate-set / oracle diagnostics
# =========================================================
def empty_candidate_oracle_stats(threshold=0.60):
    return {
        "oracle_threshold": float(threshold),
        "oracle_candidate_mean_iou": np.nan,
        "oracle_candidate_min_iou": np.nan,
        "oracle_candidate_max_iou": np.nan,
        "oracle_candidate_std_iou": np.nan,
        "oracle_candidate_variance_iou": np.nan,
        "oracle_candidate_n_frames": 0,
        "candidate_set_failure_num_failed_frames_060": np.nan,
        "candidate_set_failure_frame_ratio_060": np.nan,
        "candidate_set_failed_by_oracle_mean_060": np.nan,
        "mean_num_candidates": np.nan,
        "mean_num_candidates_iou_ge_060": np.nan,
        "mean_candidate_iou": np.nan,
        "mean_min_candidate_iou": np.nan,
        "mean_std_candidate_iou": np.nan,
        "mean_sam_gap_to_oracle": np.nan,
        "mean_b_gap_to_oracle": np.nan,
        "sam_selected_oracle_best_ratio": np.nan,
        "b_selected_oracle_best_ratio": np.nan,
    }


def candidate_oracle_stats(eval_obj, per_frame_records, threshold=0.60):
    """
    Read candidate-set/oracle diagnostics written by the run script.

    Preferred source:
      eval_summary_B_optimized.json:
        oracle_candidate_iou
        candidate_set_failure_060

    Fallback / extra source:
      eval_per_frame_B_optimized.csv or JSON per_frame records:
        oracle_iou, candidate_set_failed_060, num_candidates, etc.
    """
    out = empty_candidate_oracle_stats(threshold)

    # Summary-level fields from eval_summary_B_optimized.json.
    out["oracle_candidate_mean_iou"] = summary_stat(eval_obj, "oracle_candidate_iou", "mean")
    out["oracle_candidate_min_iou"] = summary_stat(eval_obj, "oracle_candidate_iou", "min")
    out["oracle_candidate_max_iou"] = summary_stat(eval_obj, "oracle_candidate_iou", "max")
    out["oracle_candidate_n_frames"] = safe_int(get_nested(eval_obj, "oracle_candidate_iou", "n", default=0))

    failure_obj = eval_obj.get("candidate_set_failure_060", {}) or {}
    if failure_obj:
        out["oracle_threshold"] = safe_float(failure_obj.get("threshold", threshold))
        out["candidate_set_failure_num_failed_frames_060"] = safe_float(
            failure_obj.get("num_failed_frames", np.nan)
        )
        out["candidate_set_failure_frame_ratio_060"] = safe_float(
            failure_obj.get("failure_frame_ratio", np.nan)
        )
        seq_below = failure_obj.get("sequence_oracle_mean_iou_below_060", np.nan)
        if isinstance(seq_below, bool):
            out["candidate_set_failed_by_oracle_mean_060"] = seq_below
        elif seq_below is not np.nan:
            out["candidate_set_failed_by_oracle_mean_060"] = safe_bool(seq_below)

    # Per-frame fallback / additional values.
    if not per_frame_records:
        # If summary mean exists, still derive sequence-level failure flag if missing.
        if pd.isna(out["candidate_set_failed_by_oracle_mean_060"]):
            m = safe_float(out["oracle_candidate_mean_iou"])
            out["candidate_set_failed_by_oracle_mean_060"] = bool(m < threshold) if np.isfinite(m) else np.nan
        return out

    df = pd.DataFrame(per_frame_records)

    # Oracle IoU distribution; summary only has mean/min/max/n, so std/variance come from per-frame.
    if "oracle_iou" in df.columns:
        oracle_vals = pd.to_numeric(df["oracle_iou"], errors="coerce").dropna()
        if not oracle_vals.empty:
            if not np.isfinite(out["oracle_candidate_mean_iou"]):
                out["oracle_candidate_mean_iou"] = float(oracle_vals.mean())
            if not np.isfinite(out["oracle_candidate_min_iou"]):
                out["oracle_candidate_min_iou"] = float(oracle_vals.min())
            if not np.isfinite(out["oracle_candidate_max_iou"]):
                out["oracle_candidate_max_iou"] = float(oracle_vals.max())
            if not out["oracle_candidate_n_frames"]:
                out["oracle_candidate_n_frames"] = int(oracle_vals.size)
            out["oracle_candidate_std_iou"] = float(oracle_vals.std(ddof=0))
            out["oracle_candidate_variance_iou"] = float(oracle_vals.var(ddof=0))

    # Candidate-set failure ratio per frame.
    if "candidate_set_failed_060" in df.columns:
        flags = df["candidate_set_failed_060"].apply(safe_bool)
        if len(flags) > 0:
            out["candidate_set_failure_num_failed_frames_060"] = int(flags.sum())
            out["candidate_set_failure_frame_ratio_060"] = float(flags.mean())

    # Helpful candidate statistics.
    for source_col, out_col in [
        ("num_candidates", "mean_num_candidates"),
        ("num_candidates_iou_ge_060", "mean_num_candidates_iou_ge_060"),
        ("mean_candidate_iou", "mean_candidate_iou"),
        ("min_candidate_iou", "mean_min_candidate_iou"),
        ("std_candidate_iou", "mean_std_candidate_iou"),
        ("sam_gap_to_oracle", "mean_sam_gap_to_oracle"),
        ("b_gap_to_oracle", "mean_b_gap_to_oracle"),
    ]:
        vals = numeric_series(df, source_col).dropna()
        if not vals.empty:
            out[out_col] = float(vals.mean())

    for source_col, out_col in [
        ("sam_selected_oracle_best", "sam_selected_oracle_best_ratio"),
        ("b_selected_oracle_best", "b_selected_oracle_best_ratio"),
    ]:
        if source_col in df.columns:
            flags = df[source_col].apply(safe_bool)
            if len(flags) > 0:
                out[out_col] = float(flags.mean())

    # Sequence-level candidate-set failure by oracle mean.
    m = safe_float(out["oracle_candidate_mean_iou"])
    out["candidate_set_failed_by_oracle_mean_060"] = bool(m < threshold) if np.isfinite(m) else np.nan

    return out


# =========================================================
# GT-derived target statistics
# =========================================================
def flatten_gt_target_statistics(opt_obj):
    """
    The experiment code writes gt_target_statistics with these nested keys:
      area_ratio
      C_obs_centeredness
      E_dist_border_distance_support
      edge_proximity_for_e_target
      Sil_obs
    """
    stats = opt_obj.get("gt_target_statistics", {}) or {}
    out = {
        "optimization_method": opt_obj.get("optimization_method", ""),
        "optimization_method_description": opt_obj.get("optimization_method_description", ""),
        "has_gt_target_statistics": bool(stats),
        "gt_statistic_used_for_params": stats.get("statistic_used_for_params", ""),
        "gt_stats_n_frames_used": safe_int(stats.get("n_gt_frames_used", 0)),
    }

    mapping = {
        "area_ratio": "gt_area_ratio",
        "C_obs_centeredness": "gt_C_obs",
        "E_dist_border_distance_support": "gt_E_dist",
        "edge_proximity_for_e_target": "gt_edge_proximity",
        "Sil_obs": "gt_Sil_obs",
    }

    for json_key, prefix in mapping.items():
        obj = stats.get(json_key, {}) or {}
        for stat_name in ["mean", "median", "std", "min", "max"]:
            out[f"{prefix}_{stat_name}"] = safe_float(obj.get(stat_name, np.nan))

    return out


# =========================================================
# Per-sequence collection
# =========================================================
def collect_one_sequence(seq_dir: Path, oracle_threshold=0.60):
    seq_name = seq_dir.name
    opt_path = seq_dir / "optimized_params_B.json"
    eval_path = seq_dir / "eval_summary_B_optimized.json"
    per_frame_csv_path = seq_dir / "eval_per_frame_B_optimized.csv"

    if not opt_path.exists():
        return None

    opt_obj = load_json(opt_path)

    if opt_obj.get("skipped", False):
        row = {
            "sequence": seq_name,
            "status": "skipped",
            "skip_reason": opt_obj.get("skip_reason", ""),
            "n_frames_evaluated": 0,
        }
        row.update(flatten_gt_target_statistics(opt_obj))
        row.update(per_frame_term_stats([]))
        row.update(empty_candidate_oracle_stats(oracle_threshold))
        return row

    optimized = opt_obj.get("optimized", {}) or {}
    baseline_for_optimization = opt_obj.get("baseline_for_optimization", {}) or {}

    row = {
        "sequence": seq_name,
        "status": "optimized_only" if not eval_path.exists() else "ok",
        "skip_reason": "",
        "n_total_frames_with_gt": opt_obj.get("n_total_frames_with_gt", np.nan),
        "n_used_for_opt": opt_obj.get("n_used_for_opt", np.nan),
        "gt_ratio": opt_obj.get("gt_ratio", np.nan),
        "opt_seed": opt_obj.get("opt_seed", np.nan),
        "sampling_mode": opt_obj.get("sampling_mode", ""),

        "default_train_meanIoU_for_opt": safe_float(baseline_for_optimization.get("train_meanIoU", np.nan)),
        "default_val_meanIoU_for_opt": safe_float(baseline_for_optimization.get("val_meanIoU", np.nan)),
        "opt_train_meanIoU": safe_float(optimized.get("train_meanIoU", np.nan)),
        "opt_val_meanIoU": safe_float(optimized.get("val_meanIoU", np.nan)),

        "q_border": safe_float(optimized.get("q_border", np.nan)),
        "t_area": safe_float(optimized.get("t_area", np.nan)),
        "c_target": safe_float(optimized.get("c_target", np.nan)),
        "e_target": safe_float(optimized.get("e_target", np.nan)),
        "s_target": safe_float(optimized.get("s_target", np.nan)),

        "n_frames_evaluated": 0,
        "sam_baseline_mean_iou": np.nan,
        "b_optimized_mean_iou": np.nan,
        "mean_delta_iou": np.nan,
        "best_improvement_frame": "",
        "best_improvement_delta": np.nan,
        "worst_drop_frame": "",
        "worst_drop_delta": np.nan,
    }

    row.update(flatten_gt_target_statistics(opt_obj))
    row.update(per_frame_term_stats([]))
    row.update(empty_candidate_oracle_stats(oracle_threshold))

    if not eval_path.exists():
        return row

    eval_obj = load_json(eval_path)
    row["n_frames_evaluated"] = safe_int(eval_obj.get("n_frames_evaluated", 0))
    row["sam_baseline_mean_iou"] = summary_stat(eval_obj, "baseline_iou", "mean")
    row["b_optimized_mean_iou"] = summary_stat(eval_obj, "scored_iou", "mean")
    row["mean_delta_iou"] = row["b_optimized_mean_iou"] - row["sam_baseline_mean_iou"]

    best_improve = eval_obj.get("best_improvement", {}) or {}
    worst_drop = eval_obj.get("worst_drop", {}) or {}
    row["best_improvement_frame"] = best_improve.get("key", "")
    row["best_improvement_delta"] = safe_float(best_improve.get("delta", np.nan))
    row["worst_drop_frame"] = worst_drop.get("key", "")
    row["worst_drop_delta"] = safe_float(worst_drop.get("delta", np.nan))

    # Use per-frame records inside JSON when available; fallback to CSV.
    per_frame = eval_obj.get("per_frame", [])
    if not per_frame and per_frame_csv_path.exists():
        try:
            per_frame = pd.read_csv(per_frame_csv_path).to_dict(orient="records")
        except Exception:
            per_frame = []

    row.update(per_frame_term_stats(per_frame))
    row.update(candidate_oracle_stats(eval_obj, per_frame, threshold=oracle_threshold))

    # Backward-compatible aliases used by earlier summaries.
    row["mean_score_B"] = row.get("score_B_mean", np.nan)
    row["mean_A_match"] = row.get("A_match_mean", np.nan)
    row["mean_C_match"] = row.get("C_match_mean", np.nan)
    row["mean_E_match"] = row.get("E_match_mean", np.nan)
    row["mean_S_match"] = row.get("S_match_mean", np.nan)

    return row


# =========================================================
# Global summaries
# =========================================================
def summarize_global(seq_df: pd.DataFrame, group_name="all_ok"):
    ok = seq_df[seq_df["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame([{
            "group": group_name,
            "num_sequences_ok": 0,
            "num_frames_total": 0,
        }])

    numeric_cols = [
        "n_frames_evaluated",
        "sam_baseline_mean_iou", "b_optimized_mean_iou", "mean_delta_iou",
        "oracle_candidate_mean_iou", "oracle_candidate_min_iou", "oracle_candidate_max_iou",
        "oracle_candidate_std_iou", "oracle_candidate_variance_iou",
        "candidate_set_failure_num_failed_frames_060", "candidate_set_failure_frame_ratio_060",
        "mean_num_candidates", "mean_num_candidates_iou_ge_060",
        "mean_candidate_iou", "mean_min_candidate_iou", "mean_std_candidate_iou",
        "mean_sam_gap_to_oracle", "mean_b_gap_to_oracle",
        "sam_selected_oracle_best_ratio", "b_selected_oracle_best_ratio",
        "score_B_mean", "score_B_variance", "score_B_std", "score_B_min", "score_B_max", "score_B_range",
        "A_match_mean", "A_match_variance", "A_match_std", "A_match_min", "A_match_max", "A_match_range",
        "C_match_mean", "C_match_variance", "C_match_std", "C_match_min", "C_match_max", "C_match_range",
        "E_match_mean", "E_match_variance", "E_match_std", "E_match_min", "E_match_max", "E_match_range",
        "S_match_mean", "S_match_variance", "S_match_std", "S_match_min", "S_match_max", "S_match_range",
        "t_area", "c_target", "e_target", "s_target",
    ]
    for col in numeric_cols:
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")

    weights = ok["n_frames_evaluated"].fillna(0).to_numpy(dtype=float)

    def macro(col):
        return float(ok[col].mean()) if col in ok.columns else np.nan

    def wmean(col):
        if col not in ok.columns:
            return np.nan
        vals = ok[col].to_numpy(dtype=float)
        mask = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
        if not mask.any():
            return np.nan
        return float(np.average(vals[mask], weights=weights[mask]))

    row = {
        "group": group_name,
        "num_sequences_ok": int(len(ok)),
        "num_frames_total": int(ok["n_frames_evaluated"].sum()),
        "macro_sam_baseline_mean_iou": macro("sam_baseline_mean_iou"),
        "macro_b_optimized_mean_iou": macro("b_optimized_mean_iou"),
        "macro_mean_delta_iou": macro("mean_delta_iou"),
        "weighted_sam_baseline_mean_iou": wmean("sam_baseline_mean_iou"),
        "weighted_b_optimized_mean_iou": wmean("b_optimized_mean_iou"),
        "weighted_mean_delta_iou": wmean("mean_delta_iou"),
    }

    for col in numeric_cols:
        if col == "n_frames_evaluated":
            continue
        row[f"macro_{col}"] = macro(col)
        row[f"weighted_{col}"] = wmean(col)

    return pd.DataFrame([row])


def build_global_summaries(seq_df: pd.DataFrame, oracle_threshold=0.60):
    ok = seq_df[seq_df["status"].eq("ok")].copy()
    summaries = [summarize_global(seq_df, group_name="all_ok")]

    if not ok.empty and "oracle_candidate_mean_iou" in ok.columns:
        ok["oracle_candidate_mean_iou"] = pd.to_numeric(ok["oracle_candidate_mean_iou"], errors="coerce")
        valid = ok[ok["oracle_candidate_mean_iou"] >= float(oracle_threshold)].copy()
        failed = ok[ok["oracle_candidate_mean_iou"] < float(oracle_threshold)].copy()

        summaries.append(summarize_global(valid, group_name=f"oracle_candidate_mean_iou_ge_{oracle_threshold:.2f}"))
        summaries.append(summarize_global(failed, group_name=f"oracle_candidate_mean_iou_lt_{oracle_threshold:.2f}"))

    return pd.concat(summaries, ignore_index=True)


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results_B_GT_mean",
        help="B-version result root containing sequence folders with optimized_params_B.json"
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Where to save analysis CSV files. Default: <root>/_analysis"
    )
    ap.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix tag for output filenames, e.g. gt_mean"
    )
    ap.add_argument(
        "--oracle_threshold",
        type=float,
        default=0.60,
        help=(
            "Sequence-level candidate-set failure threshold. "
            "A sequence is considered candidate-set failed if its oracle candidate mean IoU is below this value."
        )
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    output_dir = Path(args.output_dir) if args.output_dir else root / "_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    opt_files = sorted(root.rglob("optimized_params_B.json"))
    print(f"[INFO] Found {len(opt_files)} optimized_params_B.json files.")

    for opt_path in opt_files:
        row = collect_one_sequence(opt_path.parent, oracle_threshold=args.oracle_threshold)
        if row is not None:
            rows.append(row)

    if not rows:
        print("[WARN] No valid B optimized result files found.")
        return

    seq_df = pd.DataFrame(rows)

    status_order = {"ok": 0, "optimized_only": 1, "skipped": 2}
    seq_df["_status_order"] = seq_df["status"].map(status_order).fillna(99)
    seq_df = seq_df.sort_values(["_status_order", "sequence"]).drop(columns=["_status_order"])

    global_df = build_global_summaries(seq_df, oracle_threshold=args.oracle_threshold)

    suffix = f"_{args.tag}" if args.tag else ""
    per_seq_path = output_dir / f"B_optimized_per_sequence_summary_score_stats_oracle{suffix}.csv"
    global_path = output_dir / f"B_optimized_global_summary_score_stats_oracle{suffix}.csv"

    seq_df.to_csv(per_seq_path, index=False)
    global_df.to_csv(global_path, index=False)

    # Also save convenience filtered sequence lists.
    ok = seq_df[seq_df["status"].eq("ok")].copy()
    if not ok.empty and "oracle_candidate_mean_iou" in ok.columns:
        ok["oracle_candidate_mean_iou"] = pd.to_numeric(ok["oracle_candidate_mean_iou"], errors="coerce")
        valid = ok[ok["oracle_candidate_mean_iou"] >= float(args.oracle_threshold)]
        failed = ok[ok["oracle_candidate_mean_iou"] < float(args.oracle_threshold)]
        valid_path = output_dir / f"B_sequences_oracle_candidate_mean_iou_ge_{args.oracle_threshold:.2f}{suffix}.csv"
        failed_path = output_dir / f"B_sequences_oracle_candidate_mean_iou_lt_{args.oracle_threshold:.2f}{suffix}.csv"
        valid.to_csv(valid_path, index=False)
        failed.to_csv(failed_path, index=False)
        print(f"[DONE] Oracle-valid sequence list saved to: {valid_path}")
        print(f"[DONE] Oracle-failed sequence list saved to: {failed_path}")

    print(f"[DONE] Per-sequence summary saved to: {per_seq_path}")
    print(f"[DONE] Global summary saved to: {global_path}")

    print("\n=== Global summary ===")
    print(global_df.to_string(index=False))

    ok = seq_df[seq_df["status"].eq("ok")].copy()
    if not ok.empty:
        ok["mean_delta_iou"] = pd.to_numeric(ok["mean_delta_iou"], errors="coerce")
        cols = [
            "sequence",
            "n_frames_evaluated",
            "sam_baseline_mean_iou",
            "b_optimized_mean_iou",
            "mean_delta_iou",
            "oracle_candidate_mean_iou",
            "candidate_set_failure_frame_ratio_060",
            "score_B_mean",
            "score_B_min",
            "score_B_max",
            "A_match_variance",
            "C_match_variance",
            "E_match_variance",
            "S_match_variance",
        ]
        cols = [c for c in cols if c in ok.columns]

        print("\n=== Top 10 improvements ===")
        print(ok.sort_values("mean_delta_iou", ascending=False)[cols].head(10).to_string(index=False))

        print("\n=== Worst 10 drops ===")
        print(ok.sort_values("mean_delta_iou", ascending=True)[cols].head(10).to_string(index=False))

        if "oracle_candidate_mean_iou" in ok.columns:
            print("\n=== Candidate-set failures by oracle mean IoU ===")
            ok["oracle_candidate_mean_iou"] = pd.to_numeric(ok["oracle_candidate_mean_iou"], errors="coerce")
            failed = ok[ok["oracle_candidate_mean_iou"] < float(args.oracle_threshold)]
            print(failed.sort_values("oracle_candidate_mean_iou", ascending=True)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
