from models import TwoLayer, LeNet
from data import get_dataloaders
import torch

n_epochs=20
k=1
bs=128

model=LeNet(k)
dataloaders=get_dataloaders(k, bs)
loss=torch.nn.BCELoss()
optimizer=torch.optim.Adadelta(model.parameters(), lr=0.1)


for epoch in range(n_epochs):
    correct=0
    print("Epoch: ", epoch)
    for x,y in dataloaders['train']:
        out=torch.sigmoid(model(x)).reshape(-1)
        l=loss(out, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        correct+=(out.round()==y).sum()
    print("Accuracy: ", correct/len(dataloaders['train'].dataset))


torch.save(model.state_dict(), "trained_models/task1.pt")