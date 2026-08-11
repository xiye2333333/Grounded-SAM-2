import json
from pathlib import Path

p = Path(r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results_B_default\global_summary_B_480p.json")

data = json.loads(p.read_text(encoding="utf-8"))

baseline_sum = 0.0
score_sum = 0.0
n = 0

for seq_summary in data:
    # 如果 global 里同时有 optimized/default，可以只统计 default_no_gt
    if seq_summary.get("mode") != "default_no_gt":
        continue

    for rec in seq_summary["per_frame"]:
        baseline_sum += rec["baseline_iou"]
        score_sum += rec["scored_iou"]
        n += 1

print("frames:", n)
print("baseline mean IoU:", baseline_sum / n)
print("B-score mean IoU:", score_sum / n)
print("delta:", (score_sum - baseline_sum) / n)