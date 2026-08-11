import json
from pathlib import Path
import pandas as pd


# SAM_RESULTS_ROOT = Path(r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results")
SAM_RESULTS_ROOT = Path(r"D:\uwb thesis\RelatedData\train\train\SAM_results_singleobj")
OUT_CSV = SAM_RESULTS_ROOT / "summary_all_sequences.csv"
OUT_JSON = SAM_RESULTS_ROOT / "summary_all_sequences.json"

# D:\uwb thesis\RelatedData\train\train\SAM_results_singleobj*/

def safe_load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def flatten_w(w):
    # w could be list of 4 floats
    if isinstance(w, (list, tuple)) and len(w) == 4:
        return float(w[0]), float(w[1]), float(w[2]), float(w[3])
    return None, None, None, None


def main():
    if not SAM_RESULTS_ROOT.exists():
        raise SystemExit(f"Root not found: {SAM_RESULTS_ROOT}")

    rows = []
    skipped = []

    seq_dirs = sorted([p for p in SAM_RESULTS_ROOT.iterdir() if p.is_dir()])

    for seq_dir in seq_dirs:
        seq = seq_dir.name
        eval_fp = seq_dir / "eval_summary.json"
        opt_fp = seq_dir / "optimized_params.json"

        if not eval_fp.exists():
            skipped.append({"sequence": seq, "reason": "missing eval_summary.json"})
            continue

        ev = safe_load_json(eval_fp)
        if ev is None:
            skipped.append({"sequence": seq, "reason": "failed to read eval_summary.json"})
            continue

        # baseline & scored stats
        b = ev.get("baseline_iou", {}) or {}
        s = ev.get("scored_iou", {}) or {}

        # best/worst
        best = ev.get("best_improvement", {}) or {}
        worst = ev.get("worst_drop", {}) or {}

        # optimization params (optional)
        opt = safe_load_json(opt_fp) if opt_fp.exists() else None
        wA = wC = wE = wS = None
        q = t = None
        train_iou = val_iou = None
        base_train = base_val = None
        n_used_for_opt = None
        n_total_frames_with_gt = None

        if opt:
            n_used_for_opt = opt.get("n_used_for_opt")
            n_total_frames_with_gt = opt.get("n_total_frames_with_gt")

            base = (opt.get("baseline") or {})
            base_train = base.get("train_meanIoU")
            base_val = base.get("val_meanIoU")

            opt2 = (opt.get("optimized") or {})
            w = opt2.get("w")
            wA, wC, wE, wS = flatten_w(w)
            q = opt2.get("q_border")
            t = opt2.get("t_area")
            train_iou = opt2.get("train_meanIoU")
            val_iou = opt2.get("val_meanIoU")

        b_mean = b.get("mean")
        s_mean = s.get("mean")
        mean_delta = None
        if (b_mean is not None) and (s_mean is not None):
            mean_delta = float(s_mean) - float(b_mean)

        rows.append({
            "sequence": seq,

            "n_frames_evaluated": ev.get("n_frames_evaluated"),

            "baseline_mean": b.get("mean"),
            "baseline_min": b.get("min"),
            "baseline_max": b.get("max"),
            "baseline_n": b.get("n"),

            "score_mean": s.get("mean"),
            "score_min": s.get("min"),
            "score_max": s.get("max"),
            "score_n": s.get("n"),

            "mean_delta(score-baseline)": mean_delta,

            "best_improve_frame": best.get("key"),
            "best_improve_delta": best.get("delta"),
            "best_improve_baseline_iou": best.get("baseline_iou"),
            "best_improve_score_iou": best.get("scored_iou"),

            "worst_drop_frame": worst.get("key"),
            "worst_drop_delta": worst.get("delta"),
            "worst_drop_baseline_iou": worst.get("baseline_iou"),
            "worst_drop_score_iou": worst.get("scored_iou"),

            # optimization info (may be None)
            "n_total_frames_with_gt": n_total_frames_with_gt,
            "n_used_for_opt": n_used_for_opt,
            "opt_wA": wA,
            "opt_wC": wC,
            "opt_wE": wE,
            "opt_wSil": wS,
            "opt_q_border": q,
            "opt_t_area": t,
            "opt_train_meanIoU": train_iou,
            "opt_val_meanIoU": val_iou,
            "baseline_train_meanIoU": base_train,
            "baseline_val_meanIoU": base_val,

            "eval_summary_path": str(eval_fp),
            "optimized_params_path": str(opt_fp) if opt_fp.exists() else "",
        })

    if not rows:
        raise SystemExit(
            f"No sequences with eval_summary.json found under: {SAM_RESULTS_ROOT}\n"
            f"Skipped: {len(skipped)}"
        )

    df = pd.DataFrame(rows)

    # Sort: best mean_delta first, then more frames
    if "mean_delta(score-baseline)" in df.columns:
        df = df.sort_values(
            by=["mean_delta(score-baseline)", "n_frames_evaluated"],
            ascending=[False, False],
            na_position="last"
        )

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    OUT_JSON.write_text(
        json.dumps({"rows": rows, "skipped": skipped}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"[DONE] Wrote: {OUT_CSV}")
    print(f"[DONE] Wrote: {OUT_JSON}")
    print(f"Count: {len(rows)} sequences summarized | skipped: {len(skipped)}")


if __name__ == "__main__":
    main()