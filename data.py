import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
class SequenceDataset(Dataset):
    def __init__(self,data,k,transforms):
        self.transforms=transforms
        indices=torch.empty(k, len(data), dtype=torch.int64)
        for i in range(k):
            indices[i]=torch.randperm(len(data))
        self.data=torch.cat([data.data[indices[i]] for i in range(k)], dim=2)
        self.targets=sum([data.targets[indices[i]] for i in range(k)])%2

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img=self.data[idx]
        target=self.targets[idx]
        img = Image.fromarray(img.numpy(), mode="L")
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target.float()

def get_dataloaders(k, batch_size):
    data={"train": datasets.MNIST(root="data", train=True, download=True), "test": datasets.MNIST(root="data", train=False, download=True)}
    dataloaders={"train": DataLoader(SequenceDataset(data['train'], k, transforms.ToTensor()), batch_size=batch_size), "test": DataLoader(SequenceDataset(data['test'], k, transforms.ToTensor()), batch_size=batch_size)}
    return dataloaders
