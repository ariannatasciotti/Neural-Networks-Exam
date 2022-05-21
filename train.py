from models import TwoLayer, LeNet, FourLayer
from data import get_dataloaders
import torch

n_epochs=20
k=3
bs=128

model=TwoLayer(k)
dataloaders=get_dataloaders(k, bs)
loss=torch.nn.BCELoss()
#loss=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adadelta(model.parameters(), lr=0.3)


for epoch in range(n_epochs):
    train=0
    test=0
    dataloaders=get_dataloaders(k, bs)
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

torch.save(model.state_dict(), "trained_models/twolayer.pt")
