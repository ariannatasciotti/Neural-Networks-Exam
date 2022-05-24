import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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




