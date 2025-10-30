import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Read your CSV
parser = argparse.ArgumentParser(description="Visualize Synetune outputs")
parser.add_argument("--file_name", type=str, help="Path to the CSV file")
args = parser.parse_args()
file_name = args.file_name
df = pd.read_csv(file_name)

# Drop rows with NaN values in any column and print the number of dropped rows
initial_len = len(df)
df = df.dropna()
dropped_len = initial_len - len(df)
print(f"Dropped {dropped_len} rows with NaN values.")

# Define total iterations
TOTAL_ITERS = 15000

# Normalize performance values to [0,1]
perf = df["loss/val_ndcg"].values
perf_min, perf_max = perf.min(), perf.max()
# Invert alpha so that lower loss = darker
df["alpha"] = (perf - perf_min) / (perf_max - perf_min + 1e-9)

plt.figure(figsize=(10, 6))

# Plot each bell curve truncated by its progress
for _, row in df.iterrows():
    mu = row["bell_mu"]
    sigma = row["bell_sigma"]
    bell_peak = row["bell_peak"]
    alpha = row["alpha"]
    frac_limit = min(row["iter"] / TOTAL_ITERS, 1.0)

    # Generate frac_done only up to the iteration limit
    frac_done = np.linspace(0, frac_limit, 200)

    # Compute bell shape
    z = (frac_done - mu) / sigma
    shape = np.exp(-0.5 * z * z)
    closs_weight = bell_peak * shape

    # Plot in black with alpha = performance
    plt.plot(frac_done, closs_weight, color="black", alpha=alpha)

plt.xlabel("frac_done")
plt.ylabel("weight")
plt.title("Bell-shaped weight schedulers (darker = better performance)")
plt.tight_layout()
plt.savefig(f"{file_name.replace('.csv', '')}_bell_schedulers.png", dpi=300)