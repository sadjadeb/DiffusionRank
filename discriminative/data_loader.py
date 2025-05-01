import torch
import numpy as np
import random


def parse_line(line, features_count, normalization_scale):
    tokens = line.strip().split(' ')
    qid = -1
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
                feat[feat_idx - 1] = int(feat_val * normalization_scale)
            except:
                pass
    return qid, label, feat



class DataLoaderTrain():
    def __init__(self, data_file, approach='pointwise', batch_size=1024, features_count=136, normalization_scale=1000, device='cpu'):
        self.data_file = data_file
        self.approach = approach
        self.batch_size = batch_size
        self.features_count = features_count
        self.normalization_scale = normalization_scale
        self.device = device
        self.__load_data(self.data_file)

    def __iter__(self):
        self.__allocate_minibatch()
        return self

    def __load_data(self, data_file):
        self.data = {}
        with open(data_file, mode='r', encoding="utf-8") as f:
            for line in f:
                qid, label, feat = parse_line(line, self.features_count, self.normalization_scale)
                if qid not in self.data:
                    self.data[qid] = {}
                if label not in self.data[qid]:
                    self.data[qid][label] = []
                self.data[qid][label].append(feat)
        
        self.data = {k: v for k, v in self.data.items() if len(v) > 1}
        self.qids = list(self.data.keys())
    
    def __allocate_minibatch(self):
        if self.approach == 'pointwise':
            self.features = np.zeros((self.batch_size, self.features_count), dtype=np.float32)
        elif self.approach == 'pairwise':
            self.features = [np.zeros((self.batch_size, self.features_count), dtype=np.float32) for i in range(2)]
        self.labels = np.zeros((self.batch_size), dtype=np.int64)
        
    def __clear_minibatch(self):
        if self.approach == 'pointwise':
            self.features.fill(np.float32(0))
        elif self.approach == 'pairwise':
            for i in range(2):
                self.features[i].fill(np.float32(0))
            
    def __next__(self):
        self.__clear_minibatch()
        qids = random.choices(self.qids, k=self.batch_size)
        
        if self.approach == 'pointwise':
            for i in range(self.batch_size):
                labels = list(self.data[qids[i]].keys())
                label = random.choice(labels)
                feats = self.data[qids[i]][label]
                feat = feats[random.randint(0, len(feats) - 1)]
                self.labels[i] = label
                for j in range(self.features_count):
                    self.features[i, j] = feat[j] / self.normalization_scale
            return torch.from_numpy(self.features).to(self.device), torch.from_numpy(self.labels).to(self.device)  
        elif self.approach == 'pairwise':
            for i in range(self.batch_size):
                labels = random.choices(list(self.data[qids[i]].keys()), k=2)
                labels.sort(reverse=True)
                for j in range(2):
                    feats = self.data[qids[i]][labels[j]]
                    feat = feats[random.randint(0, len(feats) - 1)]
                    for k in range(self.features_count):
                        self.features[j][i, k] = feat[k] / self.normalization_scale
            return [torch.from_numpy(self.features[i]).to(self.device) for i in range(2)], torch.from_numpy(self.labels).to(self.device)


class DataLoaderTest():
    def __init__(self, data_file, batch_size=1024, features_count=136, normalization_scale=1000, device='cpu'):
        self.data_file = data_file
        self.batch_size = batch_size
        self.features_count = features_count
        self.normalization_scale = normalization_scale
        self.device = device

    def __iter__(self):
        self.reader = open(self.data_file, mode='r', encoding="utf-8")
        self.__allocate_minibatch()
        return self
    
    def __allocate_minibatch(self):
        self.features = np.zeros((self.batch_size, self.features_count), dtype=np.float32)
        self.labels = np.zeros((self.batch_size), dtype=np.int64)
        
    def __clear_minibatch(self):
        self.features.fill(np.float32(0))
            
    def __next__(self):
        self.__clear_minibatch()
        qids = []
        labels = []
        cnt = 0
        for i in range(self.batch_size):
            line = self.reader.readline()
            if line == '':
                if cnt == 0:
                    raise StopIteration
                else:
                    self.features = self.features[:cnt]
                    self.labels = self.labels[:cnt]
                    break
                
            
            qid, label, feat = parse_line(line, self.features_count, self.normalization_scale)
            qids.append(qid)
            labels.append(label)
            
            for j in range(self.features_count):
                self.features[i, j] = feat[j] / self.normalization_scale
            
            cnt += 1
        
        return torch.from_numpy(self.features).to(self.device), qids, labels, cnt
