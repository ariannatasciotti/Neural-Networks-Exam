from models import TwoLayer, LeNet, FourLayer, Classifier
from data import get_dataloaders
import torch
import csv

n_epochs=20
k=3
bs=128

model=TwoLayer(k)
#classifier=Classifier(model)

lr=0.3
optimizer1=torch.optim.AdamW(model.parameters(), lr=lr)

lambda1 = lambda epoch: 0.99 ** epoch
scheduler1=torch.optim.lr_scheduler.LambdaLR(optimizer1, lr_lambda=lambda1)

train_accuracy=list()
test_accuracy=list()

def train(model, n_epochs, optimizer, scheduler=None):
    loss=torch.nn.BCELoss()
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
        if scheduler is not None:
            scheduler.step()
        for x,y,_ in dataloaders['test']:
            out=torch.sigmoid(model(x)).reshape(-1)
            test+=(out.round()==y).sum()
        test_accuracy.append(round(test.item()/len(dataloaders['test'].dataset),2))
        train_accuracy.append(round(train.item()/len(dataloaders['train'].dataset),2))
        print("Train accuracy: ", train/len(dataloaders['train'].dataset))
        print("Test accuracy: ", test/len(dataloaders['test'].dataset))

def train_tenclass(model, n_epochs, optimizer, scheduler=None):
    loss=torch.nn.CrossEntropyLoss()
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
        if scheduler is not None:
            scheduler.step()
        for x,_,y in dataloaders['test']:
            out=classifier(x)
            test+=(torch.argmax(out,axis=1)==y).sum()
        test_accuracy.append(round(test.item()/len(dataloaders['test'].dataset),4))
        train_accuracy.append(round(train.item()/len(dataloaders['train'].dataset),4))
        print("Train accuracy: ", train/len(dataloaders['train'].dataset))
        print("Test accuracy: ", test/len(dataloaders['test'].dataset))



task='parity_task'

if task=='parity_task':
    train(model, n_epochs, optimizer1)
else:
    train_tenclass(classifier, n_epochs, optimizer1)

#path = str(model._get_name()) + "_"  + task + "_" + str(type(optimizer1).__name__) + "_" + str(type(scheduler1).__name__)
path = str(model._get_name()) + "_"  + task + "_" + str(type(optimizer1).__name__)
torch.save(model.state_dict(), "trained_models/" + str(path) + ".pt")


with open('accuracy/' + str(path) +  ".csv", "w") as f:
    writer = csv.writer(f, escapechar=' ')
    writer.writerow(('Train Accuracy', 'Test Accuracy'))
    writer.writerow([train_accuracy, test_accuracy])
