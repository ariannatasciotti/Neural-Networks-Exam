from models import TwoLayer, Classifier, FourLayer, LeNet
from data import get_dataloaders
import torch

n_epochs=20
k=3
bs=128

model=LeNet(k)
model.load_state_dict(torch.load("trained_models/LeNet.pt"))

classifier=Classifier(model)


dataloaders=get_dataloaders(k, bs)
loss=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=0.01)


for epoch in range(n_epochs):
    train=0
    test=0
    dataloaders=get_dataloaders(k, bs)
    print("Epoch: ", epoch)
    for x,_,y in dataloaders['train']:
        out=classifier(x)
        l=loss(out, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        train+=(torch.argmax(out,axis=1)==y).sum()
    for x,_,y in dataloaders['test']:
        out=classifier(x)
        test+=(torch.argmax(out,axis=1)==y).sum()
    print("Train accuracy: ", train/len(dataloaders['train'].dataset))
    print("Test accuracy: ", test/len(dataloaders['test'].dataset))
