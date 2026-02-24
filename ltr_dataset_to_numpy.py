import numpy as np
import argparse
import os


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
            except Exception as e:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LTR dataset to numpy arrays")
    parser.add_argument("--dataset", choices=["MQ2007", "MQ2008", "MSLR-WEB10K", "MSLR-WEB30K"], help="Dataset name")
    parser.add_argument("--fold", type=int, default=1, help="Fold number")
    args = parser.parse_args()

    data_base_dir = os.path.join("data", args.dataset, "raw", f"Fold{args.fold}")
    npy_base_dir = os.path.join("data", args.dataset, "npy", f"Fold{args.fold}")
    os.makedirs(npy_base_dir, exist_ok=True)
    
    features_count = 136 if "MSLR" in args.dataset else 46
    
    for split in ["train", "val", "test"]:
        data_path = os.path.join(data_base_dir, f"{split}i.txt" if split == "val" else f"{split}.txt")
        idx, labels, features = load_and_convert_to_numpy(data_path, features_count)
        np.save(os.path.join(npy_base_dir, f"idx_{split}.npy"), idx)
        np.save(os.path.join(npy_base_dir, f"y_{split}.npy"), labels)
        np.save(os.path.join(npy_base_dir, f"X_{split}.npy"), features)