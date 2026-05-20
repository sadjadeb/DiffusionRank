import numpy as np
import argparse
import os
from utils import set_all_seeds, get_features_count

set_all_seeds()


def parse_line(line, features_count):
    tokens = line.strip().split(' ')
    feat = []
    label = int(tokens[0])
    
    for i in range(features_count):
        feat.append(0)
    
    for i in range(1, len(tokens)):
        sub_tokens = tokens[i].split(':')
        if sub_tokens[0] == 'qid':
            qid = int(sub_tokens[1])
        else:
            try:
                feat_idx = int(sub_tokens[0])
                feat_val = float(sub_tokens[1])
                feat[feat_idx - 1] = feat_val
            except Exception:
                pass
    return qid, label, feat


def load_and_convert_to_numpy(data_path, features_count):
    idx = []
    labels = []
    features = []
    
    with open(data_path, mode='r', encoding="utf-8") as f:
        for line in f:
            qid, label, feat = parse_line(line, features_count)
            idx.append(qid)
            labels.append(label)
            features.append(np.array(feat))
    return np.array(idx), np.array(labels), np.array(features)


def create_subsamples(k_values, idx_train, X_train, y_train,
                      idx_val, X_val, y_val,
                      idx_test, X_test, y_test,
                      output_base_dir):
    unique_qids = np.unique(idx_train)
    qid_doc_counts = {qid: np.sum(idx_train == qid) for qid in unique_qids}

    for k in k_values:
        n_samples_target = int(len(idx_train) * k)

        shuffled_qids = np.random.permutation(unique_qids)
        selected_qids = []
        accumulated = 0
        for qid in shuffled_qids:
            if accumulated >= n_samples_target:
                break
            selected_qids.append(qid)
            accumulated += qid_doc_counts[qid]

        selected_set = set(selected_qids)
        to_select = np.array([qid in selected_set for qid in idx_train])

        idx_train_k = idx_train[to_select]
        X_train_k = X_train[to_select]
        y_train_k = y_train[to_select]

        idx_train_non_k = idx_train[~to_select]
        X_train_non_k = X_train[~to_select]
        y_train_non_k = y_train[~to_select]

        k_dir = os.path.join(output_base_dir, f"k{k}")
        os.makedirs(k_dir, exist_ok=True)

        np.save(os.path.join(k_dir, "idx_train.npy"), idx_train_k)
        np.save(os.path.join(k_dir, "X_train.npy"), X_train_k)
        np.save(os.path.join(k_dir, "y_train.npy"), y_train_k)

        np.save(os.path.join(k_dir, "idx_train_non.npy"), idx_train_non_k)
        np.save(os.path.join(k_dir, "X_train_non.npy"), X_train_non_k)
        np.save(os.path.join(k_dir, "y_train_non.npy"), y_train_non_k)

        np.save(os.path.join(k_dir, "idx_val.npy"), idx_val)
        np.save(os.path.join(k_dir, "X_val.npy"), X_val)
        np.save(os.path.join(k_dir, "y_val.npy"), y_val)

        np.save(os.path.join(k_dir, "idx_test.npy"), idx_test)
        np.save(os.path.join(k_dir, "X_test.npy"), X_test)
        np.save(os.path.join(k_dir, "y_test.npy"), y_test)

        print(f"k={k}: {len(selected_qids)}/{len(unique_qids)} qids, "
              f"{len(idx_train_k)}/{len(idx_train)} samples (target {n_samples_target}) -> {k_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LTR dataset to numpy and create fraction subsets")
    parser.add_argument("--dataset", choices=["MQ2007", "MQ2008", "MSLR-WEB10K", "MSLR-WEB30K", "Istella-S"], help="Dataset name")
    parser.add_argument("--fold", type=int, default=1, help="Fold number")
    args = parser.parse_args()

    features_count = get_features_count(args.dataset)
    data_base_dir = os.path.join("data", args.dataset, "raw", f"Fold{args.fold}")

    splits = {}
    for split in ["train", "val", "test"]:
        fname = "vali.txt" if split == "val" else f"{split}.txt"
        data_path = os.path.join(data_base_dir, fname)
        idx, labels, features = load_and_convert_to_numpy(data_path, features_count)
        splits[split] = (idx, labels, features)

    k_values = [1.0, 0.5, 0.25, 0.0625, 0.015625, 0.00390625]
    output_base_dir = os.path.join("data", args.dataset, "by_fraction", f"Fold{args.fold}")
    os.makedirs(output_base_dir, exist_ok=True)

    idx_train, y_train, X_train = splits["train"]
    idx_val, y_val, X_val = splits["val"]
    idx_test, y_test, X_test = splits["test"]

    create_subsamples(k_values, idx_train, X_train, y_train,
                      idx_val, X_val, y_val,
                      idx_test, X_test, y_test,
                      output_base_dir)
