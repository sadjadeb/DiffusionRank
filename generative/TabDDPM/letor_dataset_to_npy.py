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


if __name__ == '__main__':
    mode = 'binclass'  # ['binclass', 'multiclass']
    dataset = 'MQ2007'
    
    data_folder = os.path.join('data', dataset, 'raw', 'Fold1')
    output_folder = os.path.join('data', dataset, 'npy_bin', 'Fold1')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    train_file_path = os.path.join(data_folder, 'train.txt', mode)
    val_file_path = os.path.join(data_folder, 'vali.txt', mode)
    test_file_path = os.path.join(data_folder, 'test.txt', mode)

    # Load data
    X_train, y_train, idx_train = load_letor_data(train_file_path)
    X_val, y_val, idx_val = load_letor_data(val_file_path)
    X_test, y_test, idx_test = load_letor_data(test_file_path)

    # print shapes
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)

    # Save the data to .npy files
    np.save(os.path.join(output_folder,'idx_train.npy'), idx_train)
    np.save(os.path.join(output_folder,'idx_val.npy'), idx_val)
    np.save(os.path.join(output_folder,'idx_test.npy'), idx_test)
    np.save(os.path.join(output_folder,'X_num_train.npy'), X_train)
    np.save(os.path.join(output_folder,'X_num_val.npy'), X_val)
    np.save(os.path.join(output_folder,'X_num_test.npy'), X_test)
    np.save(os.path.join(output_folder,'y_train.npy'), y_train)
    np.save(os.path.join(output_folder,'y_val.npy'), y_val)
    np.save(os.path.join(output_folder,'y_test.npy'), y_test)
    
    # Create and save the info file
    info = {}
    
    info['name'] = dataset
    info['id'] = f'{dataset.lower()}--default'
    info['task_type'] = 'regression'
    info['n_num_features'] = X_train.shape[1]
    info['n_cat_features'] = 0
    info['train_size'] = X_train.shape[0]
    info['val_size'] = X_val.shape[0]
    info['test_size'] = X_test.shape[0]
    
    with open(os.path.join(output_folder, 'info.json'), 'w') as f:
        json.dump(info, f, indent=4)

    print(f"Data saved to {output_folder}")
