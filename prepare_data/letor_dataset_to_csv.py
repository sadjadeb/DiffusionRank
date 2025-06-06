import numpy as np
import os
import json


def load_letor_data(file_path, mode):
    # Load LETOR dataset
    # Assuming each line is formatted as: label qid:XX feature1 feature2 ... featureN #docid=XXX
    # Parse the data accordingly
    idx = []
    data = []
    labels = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[0])
            
            if mode == 'binclass':
                if label > 1:
                    label = 1
            
            features = []
            
            for i in range(1, len(parts)):
                sub_tokens = parts[i].split(':')
                if sub_tokens[0] == 'qid':
                    qid = int(sub_tokens[1])
                else:
                    try:
                        feat_idx = int(sub_tokens[0])
                        feat_val = float(sub_tokens[1])
                        features.append(feat_val)
                    except:
                        pass
            
            idx.append(qid)
            labels.append(label)
            data.append(features)
    
    return np.array(data), np.array(labels), np.array(idx)


def write_dataset_to_csv(X, y, idx, output_folder, split):
    # Save data to CSV
    with open(os.path.join(output_folder, f'{split}.csv'), 'w') as f:
        f.write(f'{",".join([str(i) for i in range(X.shape[1] + 1)])},{X.shape[1] + 1}\n')
        for i in range(X.shape[0]):
            f.write(f'{",".join([str(x) for x in X[i]])},{y[i]},{idx[i]}\n')


if __name__ == '__main__':
    mode = 'binclass'  # ['binclass', 'multiclass']
    dataset = 'MQ2007'
    
    data_folder = os.path.join('data', dataset, 'raw', 'Fold1')
    output_folder = os.path.join('generative', 'TabDiff', 'data', dataset)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    train_file_path = os.path.join(data_folder, 'train.txt')
    val_file_path = os.path.join(data_folder, 'vali.txt')
    test_file_path = os.path.join(data_folder, 'test.txt')

    # Load data
    X_train, y_train, idx_train = load_letor_data(train_file_path, mode)
    X_val, y_val, idx_val = load_letor_data(val_file_path, mode)
    X_test, y_test, idx_test = load_letor_data(test_file_path, mode)

    # Write data to CSV
    write_dataset_to_csv(X_train, y_train, idx_train, output_folder, 'train')
    write_dataset_to_csv(X_val, y_val, idx_val, output_folder, 'val')
    write_dataset_to_csv(X_test, y_test, idx_test, output_folder, 'test')
    
    print(f"Data saved to {output_folder}")
