import torch
import numpy as np
from torch import optim
import torch.nn as nn
from torchvision import transforms, models
from datasets.mydataset import MyDataset
import pandas as pd
import torch.utils.data as data
import csv
from torch.autograd import Variable
import os
from mymodels import myDenseNet, myResNet
from mymodels.utils import save_network
from torch.optim import lr_scheduler
# import adabound
from pdb import set_trace as st

exp_name = "test2"
k_fold_iter = 0
# log
log_train_file = open('./log/'+'train_'+exp_name+'.txt', 'a')
log_val_file = open('./log/'+'val_'+exp_name+'.txt', 'a')
fusion_model_name = './test2'

# super parameters
Norm = True #Normalization
BATCH_SIZE = 32
num_epoch = 30

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
exp_dir = './experiments/'+exp_name
if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir)

# transforms
transforms_train = transforms.Compose([		
	# transforms.RandomCrop(25),
	transforms.Resize(224),
])

transform_val = transforms.Compose([	
	# transforms.RandomCrop(25),
	transforms.Resize(224)
])

# datasets and dataloader
trainset = MyDataset(Istrain=0,transforms=transforms_train, Normalize=Norm, k_iter=k_fold_iter)
trainloader = data.DataLoader(trainset,batch_size=BATCH_SIZE,shuffle = True)
valset = MyDataset(Istrain=1,transforms=transform_val, Normalize=Norm, k_iter=k_fold_iter)
valloader = data.DataLoader(valset,batch_size=32,shuffle = False)
testset = MyDataset(Istrain=2,transforms=transform_val, Normalize=Norm, k_iter=k_fold_iter)
testloader = data.DataLoader(testset,batch_size = 32,shuffle = False)

# models
model = myResNet.resnet18(pretrained=False)
# model = myResNet.resnet50(pretrained=False)
# model = myDenseNet.DenseNet()
# model = myDenseNet.densenet121(pretrained=False)
model = model.cuda()

criterion = nn.CrossEntropyLoss()
# optimizer = adabound.AdaBound(model.parameters())
# optimizer = optim.Adam(model.parameters(),lr = 5e-4)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

print("Start Training...")

best_epoch = 0
best_acc = 0
for epoch in range(num_epoch):
	model.cuda()
	model.train()
	# scheduler.step()
	train_loss = 0.0
	train_acc = 0.0
    
	batch_i = 0
	for i , data in enumerate(trainloader,0):
		inputs = data['image']
		labels = data['label']
		inputs = Variable(inputs).type(torch.FloatTensor).cuda()
		labels = Variable(labels).cuda()

		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs,labels)
		loss.backward()
		optimizer.step()

		# calculate loss
		train_loss += float(loss)

		_, pred = outputs.max(1)
		num_correct = (pred==labels).sum()        
		acc1 = int(num_correct) / labels.shape[0]        
		train_acc += acc1

		batch_loss = loss.cpu().data.numpy()
		acc_log = acc1*labels.shape[0]/BATCH_SIZE

		batch_i+=1
		if batch_i %50 ==0:

			s = "epoch_"+str(epoch)+"_batch_"+str(batch_i)+"_loss_"+str(batch_loss)+"_accuracy_"+str(acc_log)+'\n'		
			log_train_file.write(s)
			log_train_file.flush()

	print("****************************")
	print("epoch: "+ str(epoch))
	print("loss: " + str(train_loss / len(trainloader)))
	print("accuracy: " + str(train_acc / len(trainloader)))
	print("****************************")

	val_acc = 0
	print("##########Validation#########")
	
	for data in valloader:
		inputs = data['image']
		labels = data['label']
		inputs = Variable(inputs).type(torch.FloatTensor).cuda()
		labels = Variable(labels).cuda()
		
		outputs = model(inputs)
		_,pred = outputs.max(1)
		num_correct = (pred==labels).sum()
		acc = int(num_correct)/labels.shape[0]
		val_acc += acc
	print("Accuracy: " + str(val_acc/len(valloader)))
	s = "epoch_"+str(epoch)+"_accuracy_"+str(val_acc/len(valloader))+"\n"
	log_val_file.write(s)
	log_val_file.flush()
	
	if val_acc/len(valloader) > best_acc:
		best_epoch=epoch
		best_acc = val_acc/len(valloader)
	print("Current Best: "+str(best_acc) + "   Best epoch: " + str(best_epoch))
	
	print("##########Validation#########")
	if(best_epoch==epoch):
		save_network(model, exp_dir, epoch)


print('Finished Training')
print('model saved at'+exp_dir)
print("Best model: " + str(best_epoch))
print("Best Accuracy: " + str(best_acc))

# test 
print("Strat Testing...")
best_model_dir = exp_dir+'/model_'+str(best_epoch)+'.pkl'
model = torch.load(best_model_dir)

result = []
model.cuda()
model.eval()
res_model = []
for data in testloader:
    inputs = data['image']
    labels = data['label']    
    inputs = Variable(inputs).type(torch.FloatTensor).cuda()
    labels = Variable(labels).cuda()
    outputs = model(inputs)
    res_model = res_model + list(np.array(outputs.detach().cpu()))

    _,pred = outputs.max(1)
    result = result + list(np.array(pred.cpu()))

# Save Fusion Material
result_np = np.array(res_model)
np.save('./results/'fusion_model_name, result_np)

# Save .csv file from sole submission(without ensemble learning)
result = np.array(result)
subm_data = pd.DataFrame(result,columns=['label'])
subm_data.to_csv('./submissions/'+exp_name+'_submit.csv')

print('Finish!')
