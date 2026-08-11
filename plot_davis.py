import matplotlib
matplotlib.use("TkAgg")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 读取你的csv/tsv文件
# 如果是逗号分隔，用 sep=","
# 如果是tab分隔，用 sep="\t"
csv_path = r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results\summary_all_sequences.csv"

df = pd.read_csv(csv_path, sep=None, engine="python")
df.columns = df.columns.str.replace('\ufeff', '')
# 确保数值列是float
df["baseline_mean"] = pd.to_numeric(df["baseline_mean"], errors="coerce")
df["score_mean"] = pd.to_numeric(df["score_mean"], errors="coerce")

# 去掉无效行
df = df.dropna(subset=["sequence", "baseline_mean", "score_mean"])

# 如果你想按原表顺序画，不排序，就不要改df
# 如果想按delta从大到小排序，取消下面这行注释
df = df.sort_values("mean_delta(score-baseline)", ascending=False)

sequences = df["sequence"].tolist()
x = np.arange(len(sequences))
width = 0.38

plt.figure(figsize=(18, 6))

plt.bar(
    x - width / 2,
    df["baseline_mean"],
    width,
    label="SAM baseline"
)

plt.bar(
    x + width / 2,
    df["score_mean"],
    width,
    label="Ours default (no GT)"
)

plt.ylabel("Mean IoU")
plt.xlabel("Sequence")
plt.title("Default training-free scoring vs SAM baseline")

plt.xticks(x, sequences, rotation=90)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()

plt.savefig("default_vs_sam_by_sequence.png", dpi=300)
plt.show()