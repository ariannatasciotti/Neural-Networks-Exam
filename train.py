from models import TwoLayer, LeNet, FourLayer, Classifier
from data import get_dataloaders
import torch
import pandas as pd





def train(model, bs, k, n_epochs, optimizer, scheduler=None):
    accuracy=dict()
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
            #hidden['epoch_' + str(epoch)]=model.extract_representation(x)
            out=torch.sigmoid(model(x)).reshape(-1)
            test+=(out.round()==y).sum()
        accuracy.setdefault('train accuracy',[]).append(round(train.item()/len(dataloaders['train'].dataset),4))
        accuracy.setdefault('test accuracy',[]).append(round(test.item()/len(dataloaders['test'].dataset),4))
        print("Train accuracy: ", train/len(dataloaders['train'].dataset))
        print("Test accuracy: ", test/len(dataloaders['test'].dataset))
        return accuracy

def train_tenclass(model, bs, k, n_epochs, optimizer, scheduler=None):
    accuracy=dict()
    loss=torch.nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        train=0
        test=0
        dataloaders=get_dataloaders(k, bs)
        print("Epoch: ", epoch)
        for x,_,y in dataloaders['train']:
            out=model(x)
            l=loss(out, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train+=(torch.argmax(out,axis=1)==y).sum()
        if scheduler is not None:
            scheduler.step()
        for x,_,y in dataloaders['test']:
            out=model(x)
            test+=(torch.argmax(out,axis=1)==y).sum()
        accuracy.setdefault('train accuracy',[]).append(round(train.item()/len(dataloaders['train'].dataset),4))
        accuracy.setdefault('test accuracy',[]).append(round(test.item()/len(dataloaders['test'].dataset),4))
        print("Train accuracy: ", train/len(dataloaders['train'].dataset))
        print("Test accuracy: ", test/len(dataloaders['test'].dataset))
    return accuracy

def main():
    n_epochs=20
    bs=128
    lr=0.01
    task='classification'
    for k in range(3,4):
        accuracy = dict()
        print(k)
        if task=='parity_task':
            model=LeNet(k)
            optimizer1=torch.optim.Adadelta(model.parameters(), lr=lr)
            train(model, bs, k, n_epochs, optimizer)
        else:
            model=FourLayer(k)
            classifier=Classifier(model, transfer=False)
            optimizer=torch.optim.Adam(classifier.parameters(), lr=lr)
            train_tenclass(classifier, bs, k, n_epochs, optimizer)


        #path = str(model._get_name()) + "_"  + task + "_" + str(type(optimizer1).__name__) + "_" + str(type(scheduler1).__name__)
        path = str(model._get_name()) + "_"  + task + "_k_"+str(k)+ "_" + str(type(optimizer).__name__)
        torch.save(model.state_dict(), "trained_models/" + str(path) + ".pt")


        df = pd.DataFrame(accuracy)
        df.to_csv('accuracy/' + str(path) +  ".csv", index=False)


if __name__ == "__main__":
    main()
