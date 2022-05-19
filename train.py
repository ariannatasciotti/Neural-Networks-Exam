from models import TwoLayer, LeNet
from data import get_dataloaders
import torch

n_epochs=20
k=3
bs=128

model=LeNet(k)
dataloaders=get_dataloaders(k, bs)
loss=torch.nn.BCELoss()
optimizer=torch.optim.Adadelta(model.parameters(), lr=0.5)


for epoch in range(n_epochs):
    train=0
    test=0
    print("Epoch: ", epoch)
    for x,y,_ in dataloaders['train']:
        out=torch.sigmoid(model(x)).reshape(-1)
        l=loss(out, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train+=(out.round()==y).sum()
    for x,y,_ in dataloaders['test']:
        out=torch.sigmoid(model(x)).reshape(-1)
        test+=(out.round()==y).sum()
    print("Train accuracy: ", train/len(dataloaders['train'].dataset))
    print("Test accuracy: ", test/len(dataloaders['test'].dataset))


torch.save(model.state_dict(), "trained_models/task1.pt")
