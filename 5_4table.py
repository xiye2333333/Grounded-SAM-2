import pandas as pd

csv_path = r"D:\uwb thesis\RelatedData\DAVIS-data\DAVIS\SAM_results\summary_all_sequences.csv"
df = pd.read_csv(csv_path)

exclude = ["bmx-bumps", "dog-agility"]

def weighted_mean(df, value_col, weight_col="score_n"):
    return (df[value_col] * df[weight_col]).sum() / df[weight_col].sum()

def make_row(name, subdf):
    default = weighted_mean(subdf, "score_mean", "score_n")
    optimized = weighted_mean(subdf, "opt_val_meanIoU", "score_n")
    return {
        "Setting": name,
        "Default training-free": default,
        "Partial-GT optimized": optimized,
        "Gain": optimized - default,
        "n_frames": int(subdf["score_n"].sum()),
        "n_sequences": int(len(subdf)),
    }

full_row = make_row("Full dataset", df)

df_excl = df[~df["sequence"].isin(exclude)].copy()
excl_row = make_row("Failure cases excluded", df_excl)

result = pd.DataFrame([full_row, excl_row])

print(result)

result.to_csv("partial_gt_weighted_table.csv", index=False)