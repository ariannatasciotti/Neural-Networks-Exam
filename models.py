import torch.nn as nn
import torch
import torch.nn.functional as F

class TwoLayer(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k=k
        self.layer1=nn.Linear(28*28*k, 512)
        self.last_layer=nn.Linear(512,1)
        self.activation=nn.ReLU()

    def forward(self, x):
        x=x.view(-1,28*28*self.k)
        x=self.activation(self.layer1(x))
        x=self.last_layer(x)
        return x

class LeNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.conv1=nn.Conv2d(1,6,5,padding=2)
        self.conv2=nn.Conv2d(6,16,5)
        self.layer1=nn.Linear(16*5*5,120)
        self.layer2=nn.Linear(120,84)
        self.last_layer=nn.Linear(84,1)
        self.activation=nn.ReLU()

    def forward(self, x):
        x=F.max_pool2d(self.activation(self.conv1(x)),2)
        x=F.max_pool2d(self.activation(self.conv2(x)),2)
        x=torch.flatten(x,1)
        x=self.activation(self.layer1(x))
        x=self.activation(self.layer2(x))
        x=self.last_layer(x)
        return x

def Classifier(parity_model):
    parity_model.last_layer=nn.Linear(parity_model.last_layer.in_features, 10)
    for name, weight in parity_model.named_parameters():
        if not name.startswith('last_layer'):
            weight.requires_grad=False
    return parity_model