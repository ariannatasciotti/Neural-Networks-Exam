import matplotlib.pyplot as plt
import pandas as pd

folder='accuracy/'
model='TwoLayer_'
task1='parity_task_'
task2='ten_class_'
optimizer1='Adadelta.csv'

model1=pd.read_csv(folder + model + task1 + optimizer1,  delimiter=',')
model2=pd.read_csv(folder + model + task2 + optimizer1,  delimiter=',')
