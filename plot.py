import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
'''
folder='accuracy/'
model='TwoLayer_'
task1='parity_task_'
task2='ten_class_'
optimizer1='Adadelta.csv'

model1=pd.read_csv(folder + model + task1 + optimizer1,  delimiter=',')
model2=pd.read_csv(folder + model + task2 + optimizer1,  delimiter=',')


plt.figure(figsize=(10,8))
plt.plot('test accuracy', data=model1, label='2L-parity task (k=3)')
plt.plot('test accuracy', data=model2, label='2L-ten class')
plt.ylabel('Accuracy',size=18)
plt.xlabel('Epoch', size=18)
#plt.title('Parity task (k=3)')
plt.legend(loc='best')
plt.ylim(0.0, 1.0)
plt.xticks(np.arange(0, 25, step=5))
plt.xlim(1, 19)
plt.show()

folder='accuracy/'
model1='TwoLayer_'
model2='FourLayer_'
model3='LeNet_'
task='parity_task_'
optimizer='Adadelta.csv'

accuracies=np.zeros((3, 4))

for i, model in enumerate([model1, model2, model3]):
    for k in range(1,5):
        data=pd.read_csv(folder+model+task+"k_"+str(k)+"_"+optimizer)
        accuracies[i,k-1]=(data['test accuracy'][19])
        #accuracies[i,k]

plt.figure(figsize=(10,8))
for i, model in enumerate([model1, model2, model3]):
    plt.plot(range(1,5), accuracies[i], 'o-', label=model[:-1])
#plt.hlines(0.5, 0.5, 4.5, colors='k')
plt.plot(range(-1,6), np.ones(7)*0.5, '-', label="Random classifier")
plt.ylabel('Accuracy',size=18)
plt.xlabel('k', size=18)
plt.legend(loc='best')
plt.ylim(0.0, 1.0)
plt.xticks(np.arange(0, 5, step=1))
plt.xlim(0.5, 4.5)
plt.savefig('accuracies_parity.png')
plt.show()


folder='accuracy/'
model1='TwoLayer_'
model2='FourLayer_'
model3='LeNet_'
task1='parity_task_'
task2='transfer_classification_'
optimizer1='Adadelta.csv'
optimizer2='Adam.csv'
k=1

accuracies=np.zeros((2, 3))
for i, model in enumerate([model1, model2, model3]):
    data=pd.read_csv(folder+model+task1+"k_"+str(k)+"_"+optimizer1)
    print(data)
    accuracies[0,i]=(data['test accuracy'][19])
    data=pd.read_csv(folder+model+task2+"k_"+str(k)+"_"+optimizer2)
    accuracies[1,i]=(data['test accuracy'][19])

plt.figure(figsize=(10,8))
plt.title("k=1", size=18)
plt.plot(range(3), accuracies[0], 'o', label='Parity task', markersize=10)
plt.plot(range(3), accuracies[1], 'o', label='Transfer learning task', markersize=10)
plt.vlines(0, accuracies[1,0], accuracies[0,0], color='k')
plt.vlines(1, accuracies[1,1], accuracies[0,1], color='k')
plt.vlines(2, accuracies[1,2], accuracies[0,2], color='k')
plt.ylabel('Accuracy',size=18)
plt.xlabel('Model', size=18)
plt.xticks(ticks=[0, 1, 2], labels=[model1[:-1], model2[:-1], model3[:-1]])
plt.legend(loc='best')
plt.savefig("transferk1.png")
plt.show()
'''


folder='accuracy/'
model='TwoLayer_'
task='parity_task_'
optimizer='Adadelta.csv'
k=3
data=pd.read_csv(folder+model+task+"k_"+str(k)+"_"+optimizer)
print(data)
accuracies=data['test accuracy']
plt.figure(figsize=(10,8))
plt.title("k=3", size=18)
plt.plot(range(1,21), accuracies, 'o-')
plt.ylim(0.5, 0.8)
plt.ylabel('Accuracy',size=18)
plt.xlabel('Epoch', size=18)
plt.xticks(ticks=range(1,21))
plt.savefig("daniely.png")
plt.show()
'''
for i, model in enumerate([model1, model2, model3]):
    data=pd.read_csv(folder+model+task1+"k_"+str(k)+"_"+optimizer1)
    print(data)
    accuracies[0,i]=(data['test accuracy'][19])
    data=pd.read_csv(folder+model+task2+"k_"+str(k)+"_"+optimizer2)
    accuracies[1,i]=(data['test accuracy'][19])

plt.figure(figsize=(10,8))
plt.title("k=1", size=18)
plt.plot(range(3), accuracies[0], 'o', label='Parity task', markersize=10)
plt.plot(range(3), accuracies[1], 'o', label='Transfer learning task', markersize=10)
plt.vlines(0, accuracies[1,0], accuracies[0,0], color='k')
plt.vlines(1, accuracies[1,1], accuracies[0,1], color='k')
plt.vlines(2, accuracies[1,2], accuracies[0,2], color='k')
plt.ylabel('Accuracy',size=18)
plt.xlabel('Model', size=18)
plt.xticks(ticks=[0, 1, 2], labels=[model1[:-1], model2[:-1], model3[:-1]])
plt.legend(loc='best')
plt.savefig("transferk1.png")
plt.show()
'''
