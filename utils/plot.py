import matplotlib.pyplot as plt
import os
from pdb import set_trace as st
import numpy as np


name = 'DenseNet121'
# train
myfile = open('../log/'+'train_final_dense121.txt')
data = myfile.readlines()
epoch_list = []
loss_list = []
acc_list = []
for d in data:
    s = d.split('\n')[0]
    s = s.split('_')
    num = int(s[1])+int(s[3])/4250
    epoch_list.append(num)
    loss_list.append(float(s[5]))
    acc_list.append(float(s[7]))
epochs = np.array(epoch_list)
losses = np.array(loss_list)
acces = np.array(acc_list)

plt.plot(epochs, losses, color='r', label='train loss')
plt.plot(epochs, acces, color='b', label='trian accuracy')

# validation
myfile = open('../log/'+'val_final_dense121.txt')
data = myfile.readlines()
val_epoch_list = []
# loss_list = []
val_acc_list = []
for d in data:
    s = d.split('\n')[0]
    s = s.split('_')
    val_epoch_list.append(int(s[1]))
    val_acc_list.append(float(s[3]))
val_epochs = np.array(val_epoch_list)
val_acces = np.array(val_acc_list)

plt.plot(val_epochs, val_acces, color='g', label='valid accuracy')


plt.xlabel('epochs')
plt.ylabel('losses and accuracy')
plt.title(name)
plt.legend()

plt.savefig('../pictures/'+name+'.png')
plt.show()
myfile.close()
