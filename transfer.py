from models import TwoLayer, Classifier, FourLayer, LeNet
from data import get_dataloaders
import torch
from train import train_tenclass
import pandas as pd

n_epochs=40
k=1
bs=128

for model in [FourLayer(k)]:
    path="trained_models/"+str(model._get_name()) + "_parity_task_k_"+str(k)+ "_Adadelta.pt"
    model.load_state_dict(torch.load(path))
    classifier=Classifier(model)
    optimizer=torch.optim.Adam(classifier.parameters(), lr=0.05)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    accuracy=train_tenclass(classifier, bs, k, n_epochs, optimizer, scheduler)

    #path = str(model._get_name()) + "_transfer_classification_k_"+str(k)+ "_" + str(type(optimizer).__name__)
    #torch.save(model.state_dict(), "trained_models/" + str(path) + ".pt")

    #df = pd.DataFrame(accuracy)
    #df.to_csv('accuracy/' + str(path) +  ".csv", index=False)
