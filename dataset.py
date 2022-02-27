from matplotlib import transforms
from spacy import load
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np

class featureDataset(Dataset):

    def __getitem__(self, index):
        with open(f'features/{index}.npy', 'rb') as f:
            ft = torch.tensor(np.load(f))
        with open(f'gtnp/{index}.npy', 'rb') as n:
            gt = torch.tensor(np.load(n))
        return(ft, gt)

    def __len__(self):
        return len(os.listdir('features'))

dataset = featureDataset()
loader = DataLoader(dataset, batch_size=16)

for batch_nxd, sample in enumerate(loader):
    print(sample)