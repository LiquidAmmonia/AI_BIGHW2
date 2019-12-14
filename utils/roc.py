import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import pandas as pd
from datasets.mydataset import MyDataset
from pdb import set_trace as st
import torch.utils.data as data
from torch.autograd import Variable
import torch
import os
import copy
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F

from torchvision import transforms, models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = 'DenseNetBC'
model_dir = './experiments/aug_right_12_SGD/model_17.pkl'
## true label
true_labels = pd.read_csv('./data/aug_data/2_val5_data.csv')


Norm = False
true_labels = np.array(true_labels)# number image_id label
true_labels = true_labels[:, 2]
true_labels = label_binarize(true_labels, np.arange(10))
# st()
## pred label
model = torch.load(model_dir)
model.cuda()
model.eval()
# st()
transform_val = transforms.Compose([	
	# transforms.RandomCrop(25),
	transforms.Resize(224)
])
valset = MyDataset(transforms=transform_val, Istrain=1, Normalize=Norm)
valloader = data.DataLoader(valset, batch_size=32, shuffle=False)
pred_labels_list = []
for data in valloader:
    inputs = data['image']
    labels = data['label']
    # st()

    inputs = Variable(inputs).type(torch.FloatTensor).cuda()
    labels = Variable(labels).cuda()
    outputs = model(inputs)
    _, pred = outputs.max(1)
    pred_one_hot = torch.sigmoid(outputs.detach().cpu())
    # st()
    pred_labels_list = pred_labels_list + list(np.array(pred_one_hot))

pred_labels = np.array(pred_labels_list)
# st()

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(10):    
    fpr[i], tpr[i], _ = metrics.roc_curve(true_labels[:, i], pred_labels[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


colors = ['b','g','r','k','c','m','y','#e24fff','#524C90','#845868']
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic: '+model_name)
# plt.plot(fpr, tpr, c =colors, lw = 2, alpha = 0.7, label = u'AUC=%.3f' % auc)
for i in range(10):
    plt.plot(fpr[i], tpr[i], label="class" + str(i) + ':ROC curve (area = %0.3f)' % roc_auc[i],color=colors[i])
plt.legend(loc="lower right")

plt.savefig('../pictures'+model_name+"_roc.png")
plt.show()
