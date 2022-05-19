import torch.nn as nn


class TwoLayer(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k=k
        self.layer1=nn.Linear(28*28*k, 512)
        self.layer2=nn.Linear(512,1)
        self.activation=nn.ReLU()
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x=x.view(-1,28*28*self.k)
        x=self.activation(self.layer1(x))
        x=self.sigmoid(self.layer2(x))
        return x.reshape(-1)
