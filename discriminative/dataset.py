import numpy as np
from torch.utils.data import Dataset


class RankingDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.labels = []
        self.qids = []

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                label = float(parts[0])
                qid = int(parts[1].split(':')[1])
                features = np.array([float(part.split(':')[1]) for part in parts[2:]], dtype=np.float32)

                self.data.append(features)
                self.labels.append(label)
                self.qids.append(qid)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.qids = np.array(self.qids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
