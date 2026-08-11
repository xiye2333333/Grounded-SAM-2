import pandas as pd
import matplotlib.pyplot as plt

csv_path = r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results\summary_all_sequences.csv"

df = pd.read_csv(csv_path)

df["gain"] = (
    df["opt_val_meanIoU"] -
    df["score_mean"]
)

df = df.sort_values("gain", ascending=False)

plt.figure(figsize=(16,5))

plt.bar(
    df["sequence"],
    df["gain"]
)

plt.axhline(0, color="black", linewidth=1)

plt.ylabel("Mean IoU gain")
plt.title("Section 5.4: Gain from partial-GT optimization")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    "partial_gt_gain.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()