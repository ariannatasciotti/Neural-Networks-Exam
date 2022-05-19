import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class SequenceDataset(Dataset):
    def __init__(self,data,k):
        indices=torch.empty(k, len(data), dtype=torch.int64)
        for i in range(k):
            indices[i]=torch.randperm(len(data))
        self.data=torch.cat([data.data[indices[i]] for i in range(k)], dim=2)
        self.targets=sum([data.targets[indices[i]] for i in range(k)])%2

    def __len__(self):
        return len(self.data)
    def __getitem(self, idx):
        return self.data[idx], self.label[idx]

def get_dataloaders(k, batch_size):
    datasets={"train": datasets.MNIST(root="data", train=True, download=True), "test": datasets.MNIST(root="data", train=False, download=True)}
    dataloaders={"train": DataLoader(SequenceDataset(datasets['train'], k), batch_size=batch_size), "test": DataLoader(SequenceDataset(datasets['test'], k), batch_size=batch_size)}
    return dataloaders