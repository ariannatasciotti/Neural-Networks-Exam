from models import TwoLayer, Classifier
from data import get_dataloaders
import torch

n_epochs=20
k=3
bs=128

model=TwoLayer(k)
model.load_state_dict(torch.load("trained_models/task1.pt"))

classifier=Classifier(model)


dataloaders=get_dataloaders(k, bs, transfer=True)
loss=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(n_epochs):
    correct=0
    print("Epoch: ", epoch)
    for x,y in dataloaders['train']:
        out=classifier(x)
        l=loss(out, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        correct+=(torch.argmax(out,axis=1)==y).sum()
    print("Accuracy: ", correct/len(dataloaders['train'].dataset))

