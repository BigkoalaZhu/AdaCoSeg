import torch
from torch.utils import data
import h5py
import random
import numpy as np
import math

class PartSpaceData(data.Dataset):
    def __init__(self, root):
        self.root = root
        f = h5py.File(root, 'r')
        self.wholeshapes = f['wholeshapes']
        self.labels = f['labels']
        self.parts = f['parts']

    def __getitem__(self, index):
        shape = torch.from_numpy(self.wholeshapes[index]).float()
        l = torch.from_numpy(np.squeeze(self.labels[index])).long()
        p = torch.from_numpy(np.squeeze(self.parts[index])).long()

        return shape, p, l

    def __len__(self):
        return self.wholeshapes.len()

class ShapeSegDataTest(data.Dataset):
    def __init__(self, root):
        self.root = root
        f = h5py.File(root, 'r')
        wholeshapes = f['wholeshapes']
        parts = f['parts']
        self.index = random.sample(range(wholeshapes.len()), 12)
        self.wholeshapes = [wholeshapes[i] for i in self.index]
        self.parts = [parts[i] for i in self.index]

    def __getitem__(self, index):
        shape = torch.from_numpy(self.wholeshapes[index]).float()
        p = torch.from_numpy(np.squeeze(self.parts[index])).long()
        return shape, p

    def getIndex(self):
        return self.index

    def __len__(self):
        return len(self.wholeshapes)