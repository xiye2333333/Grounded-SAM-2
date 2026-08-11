import json
import math
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd


# =========================
# Config
# =========================

ROOT_DIR = Path(r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results")
CSV_PATH = ROOT_DIR / "summary_all_sequences.csv"

JSON_NAME = "eval_summary_optimized.json"

# Numerical tolerance for float comparison
TOL = 1e-8

# If True, overwrite the CSV after correction.
# If False, only generate report without modifying CSV.
APPLY_FIX = True


# =========================
# Helper functions
# =========================

def is_missing(x):
    """Return True if x is NaN/None/empty."""
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


def values_different(old, new, tol=TOL):
    """Compare old CSV value and new JSON value."""
    if is_missing(old):
        return True

    try:
        old_f = float(old)
        new_f = float(new)
        return abs(old_f - new_f) > tol
    except Exception:
        return str(old) != str(new)


def get_nested(d, keys, default=None):
    """Safely get nested dictionary value."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def find_train_mean_iou(data):
    """
    Try to find train mean IoU from possible JSON structures.
    The example JSON does not contain this value, so this function may return None.
    """
    possible_paths = [
        ["train_iou", "mean"],
        ["train_scored_iou", "mean"],
        ["optimized_train_iou", "mean"],
        ["opt_train_iou", "mean"],
        ["train_meanIoU"],
        ["opt_train_meanIoU"],
    ]

    for path in possible_paths:
        value = get_nested(data, path, default=None)
        if value is not None:
            return value

    return None


# =========================
# Main checking logic
# =========================

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    if "sequence" not in df.columns:
        raise ValueError("CSV must contain a 'sequence' column.")

    # Ensure target columns exist.
    required_columns = [
        "opt_wA",
        "opt_wC",
        "opt_wE",
        "opt_wSil",
        "opt_q_border",
        "opt_t_area",
        "opt_val_meanIoU",
    ]

    for col in required_columns:
        if col not in df.columns:
            print(f"[WARN] Column '{col}' not found in CSV. Creating it.")
            df[col] = pd.NA

    report_rows = []
    missing_json = []
    sequence_mismatch = []
    skipped_train_check = []

    checked_count = 0
    fixed_count = 0

    for idx, row in df.iterrows():
        seq = str(row["sequence"]).strip()
        seq_dir = ROOT_DIR / seq
        json_path = seq_dir / JSON_NAME

        if not json_path.exists():
            missing_json.append(seq)
            report_rows.append({
                "sequence": seq,
                "field": "__json__",
                "csv_value": "",
                "json_value": "",
                "status": "MISSING_JSON",
                "json_path": str(json_path),
            })
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        checked_count += 1

        json_seq = data.get("sequence", None)
        if json_seq is not None and str(json_seq).strip() != seq:
            sequence_mismatch.append((seq, json_seq))
            report_rows.append({
                "sequence": seq,
                "field": "__sequence__",
                "csv_value": seq,
                "json_value": json_seq,
                "status": "SEQUENCE_MISMATCH",
                "json_path": str(json_path),
            })

        mode = data.get("mode", None)
        if mode is not None and mode != "optimized":
            report_rows.append({
                "sequence": seq,
                "field": "__mode__",
                "csv_value": "",
                "json_value": mode,
                "status": "WARNING_NOT_OPTIMIZED_MODE",
                "json_path": str(json_path),
            })

        params = data.get("params", {})
        w = params.get("w", None)

        if not isinstance(w, list) or len(w) != 4:
            report_rows.append({
                "sequence": seq,
                "field": "params.w",
                "csv_value": "",
                "json_value": str(w),
                "status": "INVALID_WEIGHT_IN_JSON",
                "json_path": str(json_path),
            })
            continue

        json_values = {
            "opt_wA": w[0],
            "opt_wC": w[1],
            "opt_wE": w[2],
            "opt_wSil": w[3],
            "opt_q_border": params.get("q_border", None),
            "opt_t_area": params.get("t_area", None),
            "opt_val_meanIoU": get_nested(data, ["scored_iou", "mean"], default=None),
        }

        # opt_train_meanIoU: check only if CSV has this column and JSON contains a recognizable train mean.
        if "opt_train_meanIoU" in df.columns:
            train_mean = find_train_mean_iou(data)
            if train_mean is not None:
                json_values["opt_train_meanIoU"] = train_mean
            else:
                skipped_train_check.append(seq)
                report_rows.append({
                    "sequence": seq,
                    "field": "opt_train_meanIoU",
                    "csv_value": row.get("opt_train_meanIoU", ""),
                    "json_value": "",
                    "status": "SKIPPED_NO_TRAIN_MEAN_IN_JSON",
                    "json_path": str(json_path),
                })

        for field, json_value in json_values.items():
            if json_value is None:
                report_rows.append({
                    "sequence": seq,
                    "field": field,
                    "csv_value": row.get(field, ""),
                    "json_value": "",
                    "status": "MISSING_FIELD_IN_JSON",
                    "json_path": str(json_path),
                })
                continue

            csv_value = row.get(field, pd.NA)

            if values_different(csv_value, json_value):
                report_rows.append({
                    "sequence": seq,
                    "field": field,
                    "csv_value": csv_value,
                    "json_value": json_value,
                    "status": "FIXED" if APPLY_FIX else "DIFFERENT_NOT_FIXED",
                    "json_path": str(json_path),
                })

                if APPLY_FIX:
                    df.at[idx, field] = json_value
                    fixed_count += 1
            else:
                report_rows.append({
                    "sequence": seq,
                    "field": field,
                    "csv_value": csv_value,
                    "json_value": json_value,
                    "status": "OK",
                    "json_path": str(json_path),
                })

    # Check if there are sequence folders with optimized JSON but not in CSV.
    csv_sequences = set(str(x).strip() for x in df["sequence"].dropna().tolist())
    extra_json_folders = []

    for child in ROOT_DIR.iterdir():
        if child.is_dir() and (child / JSON_NAME).exists():
            if child.name not in csv_sequences:
                extra_json_folders.append(child.name)
                report_rows.append({
                    "sequence": child.name,
                    "field": "__folder__",
                    "csv_value": "",
                    "json_value": str(child / JSON_NAME),
                    "status": "JSON_EXISTS_BUT_SEQUENCE_NOT_IN_CSV",
                    "json_path": str(child / JSON_NAME),
                })

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = ROOT_DIR / f"summary_check_report_{timestamp}.csv"
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    # Save corrected CSV
    if APPLY_FIX:
        backup_path = ROOT_DIR / f"summary_all_sequences_backup_before_fix_{timestamp}.csv"
        shutil.copy2(CSV_PATH, backup_path)

        df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

        print(f"[DONE] CSV checked and corrected.")
        print(f"[BACKUP] Original CSV backed up to:")
        print(f"  {backup_path}")
        print(f"[UPDATED] Corrected CSV saved to:")
        print(f"  {CSV_PATH}")
    else:
        print(f"[DONE] CSV checked. No modification was applied because APPLY_FIX=False.")

    print()
    print("========== Summary ==========")
    print(f"Sequences checked with JSON: {checked_count}")
    print(f"Fields fixed: {fixed_count}")
    print(f"Missing JSON files: {len(missing_json)}")
    print(f"Sequence-name mismatches: {len(sequence_mismatch)}")
    print(f"Extra JSON folders not in CSV: {len(extra_json_folders)}")
    print(f"Train IoU checks skipped: {len(skipped_train_check)}")
    print()
    print(f"[REPORT] Full check report saved to:")
    print(f"  {report_path}")

    if missing_json:
        print()
        print("Missing JSON sequences:")
        for seq in missing_json:
            print(f"  - {seq}")

    if sequence_mismatch:
        print()
        print("Sequence mismatch:")
        for csv_seq, json_seq in sequence_mismatch:
            print(f"  - CSV: {csv_seq} | JSON: {json_seq}")

    if extra_json_folders:
        print()
        print("Folders with JSON but not in CSV:")
        for seq in extra_json_folders:
            print(f"  - {seq}")


if __name__ == "__main__":
    main()