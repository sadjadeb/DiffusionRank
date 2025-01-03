import numpy as np
import os


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


if __name__ == '__main__':
    dataset = 'MSLR-WEB30K'
    
    data_folder = os.path.join('data', dataset, 'raw', 'Fold1')
    output_folder = os.path.join('data', dataset, 'npy', 'Fold1')
    
    train_file_path = os.path.join(data_folder, 'train.txt')
    val_file_path = os.path.join(data_folder, 'vali.txt')
    test_file_path = os.path.join(data_folder, 'test.txt')

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

    print(f"Data saved to {output_folder}")
