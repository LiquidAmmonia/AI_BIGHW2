import numpy as np
from pdb import set_trace as st
import pandas as pd
import torch.nn.functional as F
from torchvision import transforms
import torch
import os
exp_name = 'test'

model_list = []
for model_name in os.listdir('./results/'):
    model_list.append(np.load('./results/'+model_name))

result = []
for i in range(5000):
    model_fusion = 0
    for model in model_list:
        model_fusion+=model[i]
    pred = model_fusion.argmax()
    result.append(pred)

result = np.array(result)
subm_data = pd.DataFrame(result,columns=['label'])
subm_data.to_csv('./submissions/'+exp_name+'_submit.csv')
print('Finish writing')
