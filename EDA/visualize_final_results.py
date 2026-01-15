import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def format_k(x, pos):
    """Format numbers as 'k' (thousands) for readability."""
    if x >= 1000:
        return f'{x/1000:.0f}k'
    return f'{x:.0f}'


# Training samples per dataset (at k=1.0)
TRAIN_SAMPLES = {
    "MQ2007": 42158,
    "MQ2008": 9630,
    "MSLR-WEB10K": 723412,
}


def plot_method(x_values, pivot, method, dataset, color, marker, linestyle):
    """Plot a single method line if it exists in the pivot table."""
    if method in pivot.columns:
        plt.plot(
            # pivot.index, pivot[method],
            x_values, pivot[method],
            marker=marker, linestyle=linestyle, linewidth=2,
            label=f"{dataset} — {method}", color=color
        )


# Load and reshape performance data
csv_path = "Gen_LTR - RQ1.csv"

raw = pd.read_csv(csv_path, header=None, dtype=str).fillna('')

# Build column names from two-row header
header_row0 = raw.iloc[0].tolist()
header_row1 = raw.iloc[1].tolist()

cols = []
for a, b in zip(header_row0, header_row1):
    a, b = a.strip(), b.strip()
    if a == '' and b == '':
        cols.append('idx')
    elif a and b:
        cols.append(f"{a} {b}")
    else:
        cols.append(a or b)

data = raw.iloc[2:].copy()
data.columns = cols

# Parse rows: detect 'K...' rows as section markers and method rows as data
records = []
current_k = None
ndcg_cols = [c for c in data.columns if "NDCG@10" in c]

for _, row in data.iterrows():
    idx_val = row['idx'].strip()
    
    if idx_val.upper().startswith('K'):
        try:
            current_k = float(idx_val.lstrip('Kk'))
        except ValueError:
            current_k = None
        continue
    
    method = idx_val
    if not method:
        continue
    
    for col in ndcg_cols:
        dataset = col.replace(" NDCG@10", "").strip()
        val = row[col].strip()
        if not val:
            continue
        try:
            ndcg = float(val)
            records.append({"dataset": dataset, "k": current_k, "method": method, "ndcg": ndcg})
        except ValueError:
            continue

tidy = pd.DataFrame.from_records(records)

# Filter only the methods we want
tidy = tidy[tidy['method'].isin(['Ours', 'Discriminative', 'Ours - only-labeled', 'Ours - w-unlabeled'])].copy()
tidy = tidy.dropna(subset=['k'])

# Prepare plotting
datasets = sorted(tidy['dataset'].unique())
color_map = {ds: plt.cm.tab10(i / max(1, len(datasets) - 1)) for i, ds in enumerate(datasets)}
unique_k_sorted = sorted(tidy['k'].unique())

plt.figure(figsize=(8, 5))
plt.xscale('log')

for ds in datasets:
    sub = tidy[tidy['dataset'] == ds]
    pivot = sub.pivot_table(index='k', columns='method', values='ndcg', aggfunc='mean')
    pivot = pivot.reindex(unique_k_sorted)
    
    # Calculate actual training samples for x-axis
    x_values = [k * TRAIN_SAMPLES[ds] for k in pivot.index]
    
    if 'RQ1' in csv_path:
        plot_method(x_values, pivot, 'Ours', ds, color_map[ds], 'o', '-')
    elif 'RQ2' in csv_path:
        plot_method(x_values, pivot, 'Ours - only-labeled', ds, color_map[ds], 'o', '-')
        plot_method(x_values, pivot, 'Ours - w-unlabeled', ds, color_map[ds], '*', ':')
    
    plot_method(x_values, pivot, 'Discriminative', ds, color_map[ds], 's', '--')

# Formatting
plt.gca().xaxis.set_major_formatter(FuncFormatter(format_k))
plt.xlabel("# Training Samples")
plt.ylabel("NDCG@10")
plt.title("DiffusionRank vs. Discriminative")
# plt.xticks(unique_k_sorted, [str(k) for k in unique_k_sorted], rotation=45)
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.legend(title="Dataset & Method", loc="lower right", fontsize='small', title_fontsize='small')
plt.tight_layout()

plt.savefig("ndcg_across_datasets_x_axis_train_samples.png", dpi=300)
