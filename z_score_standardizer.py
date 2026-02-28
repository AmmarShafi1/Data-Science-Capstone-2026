import pandas as pd

# =========================
# Load Data
# =========================
INPUT_FILE = "ai_intensity_by_ces_bucket.csv"
OUTPUT_FILE = "ai_intensity_by_ces_bucket_scaled.csv"

df = pd.read_csv(INPUT_FILE)

# =========================
# Standardize (Z-Score)
# =========================
mean_val = df["ai_intensity"].mean()
std_val = df["ai_intensity"].std()

df["ai_intensity_zscore"] = (
    df["ai_intensity"] - mean_val
) / std_val

# =========================
# Save Output
# =========================
df.to_csv(OUTPUT_FILE, index=False)

print("Mean AI intensity:", mean_val)
print("Std Dev AI intensity:", std_val)
print("\nScaled values:")
print(df)