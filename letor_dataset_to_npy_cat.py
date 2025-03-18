import numpy as np
import os
import json


def load_letor_data(file_path):
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

def parse_line(line):
    tokens = line.strip().split(' ')
    qid = -1
    feat = []
    label = int(tokens[0])
    
    for i in range(46):
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
            except:
                pass
    return qid, label, feat


def find_and_cast_to_cat(dataset, X_all_num, y):
    X_num = X_all_num.copy()
    
    if dataset == 'MQ2007':
        # Drop feature no. 46 as it is always zero except for one instance
        X_num = np.delete(X_num, 45, axis=1)
    
    # Drop features with zero variance
    std = np.std(X_num, axis=0)
    X_num = X_num[:, std != 0]
    
    # Find the categorical features
    # cat_features = []
    # for i in range(X_num.shape[1]):
    #     if len(np.unique(X_num[:,i])) < 5:
    #         cat_features.append(i)
    
    cat_features = []
    # Cast labels to categorical by makeing them str
    cat_features.append(y.astype(str))
    
    X_cat = np.array(cat_features)
    X_cat = X_cat.T
    
    print(f"Number of categorical features: {X_cat.shape[1]}")
    print(f"Number of numerical features: {X_num.shape[1]}")
    
    return X_num, X_cat, y


if __name__ == '__main__':
    dataset = 'MSLR-WEB30K'
    
    data_folder = os.path.join('data', dataset, 'raw', 'Fold1')
    output_folder = os.path.join('data', dataset, 'npy_cat', 'Fold1')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    train_file_path = os.path.join(data_folder, 'train.txt')
    val_file_path = os.path.join(data_folder, 'vali.txt')
    test_file_path = os.path.join(data_folder, 'test.txt')

    # Load data
    X_train, y_train, idx_train = load_letor_data(train_file_path)
    X_val, y_val, idx_val = load_letor_data(val_file_path)
    X_test, y_test, idx_test = load_letor_data(test_file_path)

    
    # Cast to categorical
    X_num_train, X_cat_train, y_train = find_and_cast_to_cat(dataset, X_train, y_train)
    X_num_val, X_cat_val, y_val = find_and_cast_to_cat(dataset, X_val, y_val)
    X_num_test, X_cat_test, y_test = find_and_cast_to_cat(dataset, X_test, y_test)

    # print shapes
    print(X_num_train.shape, X_cat_train.shape, y_train.shape)
    print(X_num_val.shape, X_cat_val.shape, y_val.shape)
    print(X_num_test.shape, X_cat_test.shape, y_test.shape)
    
    
    # Save the data to .npy files
    np.save(os.path.join(output_folder,'idx_train.npy'), idx_train)
    np.save(os.path.join(output_folder,'idx_val.npy'), idx_val)
    np.save(os.path.join(output_folder,'idx_test.npy'), idx_test)
    np.save(os.path.join(output_folder,'X_num_train.npy'), X_num_train)
    np.save(os.path.join(output_folder,'X_num_val.npy'), X_num_val)
    np.save(os.path.join(output_folder,'X_num_test.npy'), X_num_test)
    np.save(os.path.join(output_folder,'X_cat_train.npy'), X_cat_train)
    np.save(os.path.join(output_folder,'X_cat_val.npy'), X_cat_val)
    np.save(os.path.join(output_folder,'X_cat_test.npy'), X_cat_test)
    np.save(os.path.join(output_folder,'y_train.npy'), y_train)
    np.save(os.path.join(output_folder,'y_val.npy'), y_val)
    np.save(os.path.join(output_folder,'y_test.npy'), y_test)
    
    # Create and save the info file
    info = {}
    
    info['name'] = dataset
    info['id'] = f'{dataset.lower()}--cat'
    info['task_type'] = 'multiclass'
    info['n_classes'] = len(np.unique(y_train))
    info['n_num_features'] = X_num_train.shape[1]
    info['n_cat_features'] = X_cat_train.shape[1]
    info['train_size'] = X_num_train.shape[0]
    info['val_size'] = X_num_val.shape[0]
    info['test_size'] = X_num_test.shape[0]
    
    with open(os.path.join(output_folder, 'info.json'), 'w') as f:
        json.dump(info, f, indent=4)

    print(f"Data saved to {output_folder}")
