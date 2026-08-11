import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results\summary_all_sequences.csv")

selected_sequences = [
    "car-shadow",
    "breakdance",
    "kite-surf",
    "drift-straight",
    "lucia",
    "libby",
    "dog-agility",
    "dance-jump",
    "bmx-bumps",
    "mallard-fly",
    "car-roundabout",
    "camel",
    "mallard-water",
    "boat",
    "dance-twirl",
    "drift-chicane",
    "hike",
    "breakdance-flare",
    "bmx-trees",
    "goat",
    "paragliding",
    "horsejump-high",
    "elephant",
    "dog",
    "horsejump-low",
    "cows",
    "blackswan",
    "bus",
    "hockey",
    "bear",
    "kite-walk"
]

# 保留CSV中实际存在的序列
selected_sequences = [
    s for s in selected_sequences
    if s in df["sequence"].values
]

plot_df = (
    df.set_index("sequence")
      .loc[selected_sequences]
      .reset_index()
)

x = np.arange(len(plot_df))
width = 0.4

plt.figure(figsize=(18, 5))

plt.bar(
    x - width/2,
    plot_df["score_mean"],
    width,
    label="Default weight"
)

plt.bar(
    x + width/2,
    plot_df["opt_val_meanIoU"],
    width,
    label="Optimized weight"
)

plt.xticks(
    x,
    plot_df["sequence"],
    rotation=90
)

plt.ylabel("Mean IoU")
plt.xlabel("Sequence")
plt.title("Default weights vs optimized weights")
plt.legend()

plt.tight_layout()
plt.savefig(
    "default_vs_optimized_weights.png",
    dpi=300,
    bbox_inches="tight"
)