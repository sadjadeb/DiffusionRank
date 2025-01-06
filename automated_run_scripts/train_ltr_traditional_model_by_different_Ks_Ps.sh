#!/bin/bash

# Sub-experiment number
sub=2

# Array of k_values
k_values=(1.0 0.25 0.0625)

# Array of p_values
p_values=(0.0 0.25 0.5 0.75 1.0)

# Iterate over the k_values and p_values
for k in "${k_values[@]}"; do
    for p in "${p_values[@]}"; do
        echo "Running for k=$k and p=$p (sub=$sub)"
        python main_traditional.py --p "$p" --k "$k" --sub "$sub"
    done
done
