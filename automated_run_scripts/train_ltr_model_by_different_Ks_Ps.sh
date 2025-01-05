#!/bin/bash

# Array of k_values
k_values=(1.0 0.5 0.25 0.125 0.0625)

# Array of p_values
p_values=(0.0 0.2 0.4 0.6 0.8 1.0)

# Iterate over the k_values and p_values
for k in "${k_values[@]}"; do
    for p in "${p_values[@]}"; do
        echo "Running for k=$k and p=$p"
        python main.py --p "$p" --k "$k"
    done
done
