import pandas as pd

# Load evaluation results
csv_path = "evaluation_scores_vocos_mel_24khz.csv"
df = pd.read_csv(csv_path)

# Filter rows where Audio_ID starts with 'dev-clean'
dev_clean_df = df[df["Audio_ID"].str.startswith("dev-clean")]

# Select numerical evaluation metric columns
metric_cols = ["UTMOS", "PESQ", "V/UV F1", "Periodicity", "M-STFT", "Pitch", "MCD"]
metric_cols = [col for col in metric_cols if col in dev_clean_df.columns]

# Compute mean scores
dev_clean_means = dev_clean_df[metric_cols].mean()

print("=== Dev-Clean Metrics Summary ===")
print(dev_clean_means.to_string())