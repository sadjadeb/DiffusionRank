#!/bin/bash

# Array of k_values
k_values=(1.0 0.25 0.0625)

# Iterate over the k_values
for k in "${k_values[@]}"; do
    echo "Running for k=$k"
    python scripts/train.py --config exp/MQ2007/sajad_config.toml --k "$k"
done
