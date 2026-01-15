from scipy import stats
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_file1", type=str, help="Path to the run file", required=True)
parser.add_argument("--run_file2", type=str, help="Path to the run file", required=True)
args = parser.parse_args()

with open(args.run_file1) as f1, open(args.run_file2) as f2:
    r1_lines = f1.readlines()
    r2_lines = f2.readlines()

    r1_preds = [float(line.strip().split()[2]) for line in r1_lines[1:]]
    r2_preds = [float(line.strip().split()[2]) for line in r2_lines[1:]]


# Perform the t-test
t_stat, p_value = stats.ttest_ind(r1_preds, r2_preds)
print("T-statistic:", t_stat)
print("P-value:", p_value)

# Significance level
alpha = 0.05
if p_value <= alpha:
    print("The null hypothesis is rejected. The two runs have significantly different predictions.")
else:
    print("The null hypothesis is not rejected. The two runs do not have significantly different predictions.")
