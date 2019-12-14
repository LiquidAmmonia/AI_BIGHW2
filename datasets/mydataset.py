import torch
import numpy as np
from torch import optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import pandas as pd
from PIL import Image
from pdb import set_trace as st

class MyDataset(data.Dataset):
    def __init__(self, transforms = None,Istrain = 0, Normalize = False, k_iter=0):
        super(MyDataset, self).__init__()
        if Istrain==0:
            self.imgs = np.load('./data/aug_data/train5.npy')
            self.labels = pd.read_csv('./data/aug_data/'+str(k_iter)+'_train5_data.csv')
            print("Number of Images in Trainset: "+ str(self.labels.shape[0]))
            
        elif Istrain==1:
            self.imgs = np.load('./data/aug_data/train5.npy')
            self.labels = pd.read_csv('./data/aug_data/'+str(k_iter)+'_val5_data.csv')
            print("Number of Images in Valset: "+ str(self.labels.shape[0]))
        else:
            self.imgs = np.load('./data/original_data/test.npy')
            self.labels = pd.read_csv('./data/original_data/samplesummission.csv')
        self.transforms = transforms
        self.Normalize = Normalize
        

    def __getitem__(self,index):
        img_id = self.labels.loc[index]['image_id']
        label = self.labels.loc[index]['label']
        
        img = self.imgs[img_id]
        img = np.reshape(img,(28,28))
        img = Image.fromarray(img)
        
        if self.transforms:
            img = self.transforms(img)
        
        img = transforms.ToTensor()(img)
        if self.Normalize:
            img = transforms.Normalize([0.5], [0.5])(img)
        sample = {'image':img,'label':label}
        return sample        

    def __len__(self):
        return len(self.labels)