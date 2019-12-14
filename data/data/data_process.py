# Data Augmentation

## input: Original data
## Output: Augmentation Result: train5.npy and 
#          k-folded sets: {i}_train5_data.csv {i}_val5_data.csv i = 0,1,2,3,4

import pandas as pd 
import numpy as np
from pdb import set_trace as st
from PIL import Image, ImageFilter
from torchvision import transforms
import os

if not os.path.exists('./aug_data'):
    os.mkdir('./aug_data')
b = 30000

result = np.zeros((30000*5, 784), dtype=int)
ori_imgs = np.load('./original_data/train.npy')

hori = np.zeros_like(ori_imgs)
result[0:b, :] = ori_imgs
# Horizontal Flip
for i in range(len(ori_imgs)):
    img = ori_imgs[i].reshape(28, 28)
    img = Image.fromarray(img)
    img = transforms.RandomHorizontalFlip(p=1)(img)
    img = np.array(img).reshape(784)
    hori[i]=img
result[b:2*b] = hori

print("finish 1")
# Add Noise
noi = np.zeros_like(ori_imgs)
for i in range(len(ori_imgs)):
    img = ori_imgs[i].reshape(28, 28)
    noise_num = 10
    noise = np.random.randint(0, 255, noise_num)
    rand_x_index = np.random.randint(0, 27, noise_num)
    rand_y_index = np.random.randint(0, 27, noise_num)
    img[rand_x_index, rand_y_index] = noise
    img = img.reshape(784)
    noi[i]=img
result[2*b:3*b] = noi

print("finish 2")
# Random Crop
crop = np.zeros_like(ori_imgs)
for i in range(len(ori_imgs)):
    img = ori_imgs[i].reshape(28, 28)
    img = Image.fromarray(img)
    img = transforms.RandomCrop(15)(img)
    img = transforms.Resize(28)(img)
    img = np.array(img).reshape(784)
    crop[i]=img
result[3*b:4*b] = crop


print("finish 3")
# Blur
blur = np.zeros_like(ori_imgs)
for i in range(len(ori_imgs)):
    img = ori_imgs[i].reshape(28, 28)
    img = Image.fromarray(img)
    img = img.filter(ImageFilter.BLUR) 
    img = np.array(img).reshape(784)
    blur[i]=img
result[4*b:5*b] = blur
result = result.astype(np.uint8)
np.save('./aug_data/train5.npy', result)
print("finish writting train5.npy")

#get validation set
data = pd.read_csv('./original_data/train.csv')
data = data.sample(frac=1.0)
data = data.reset_index()
data = data.drop(columns='index')

val_data = data.loc[0:2999]
train_data = data.loc[3000:]

val_data_np = np.array(val_data)
train_data_np = np.array(train_data)

fmt1 = np.zeros_like(val_data_np)
fmt1[:, 0] = b

fmt2 = np.zeros_like(train_data_np)
fmt2[:, 0] = b

# val_data_np = np.vstack((val_data_np, val_data_np + fmt1, val_data_np + 2 * fmt1, val_data_np + 3 * fmt1, val_data_np + 4 * fmt1))
train_data_np = np.vstack((train_data_np, train_data_np + fmt2, train_data_np + 2 * fmt2, train_data_np + 3 * fmt2, train_data_np + 4 * fmt2))

# # 真实数据
# df1 = pd.DataFrame()
# df1["image_id"] = val_data_np[:, 0]
# df1["label"] = val_data_np[:, 1]
# df1.to_csv('./val5_data.csv')

# df2 = pd.DataFrame()
# df2["image_id"] = train_data_np[:, 0]
# df2["label"] = train_data_np[:, 1]

# df2.to_csv('./train5_data.csv')



# data_np = np.array(data)
# data_np = np.vstack((data_np, data_np, data_np, data_np, data_np))
# df = pd.DataFrame()
# df["image_id"] = data_np[:, 0]
# df["label"] = data_np[:, 1]
# df.to_csv('./train5.csv')

# st()
#get validation set
data = pd.read_csv('./original_data/train.csv')
data = data.sample(frac=1.0)
data = data.reset_index()
data = data.drop(columns='index')

for i in range(5):
    
    val_data = data.loc[i*3000:(i+1)*3000-1]
    train_data = np.vstack((data.loc[3000*(i+1):], data.loc[0:i*3000-1]))

    val_data_np = np.array(val_data)
    train_data_np = np.array(train_data)

    fmt1 = np.zeros_like(val_data_np)
    fmt1[:, 0] = b

    fmt2 = np.zeros_like(train_data_np)
    fmt2[:, 0] = b

    # val_data_np = np.vstack((val_data_np, val_data_np + fmt1, val_data_np + 2 * fmt1, val_data_np + 3 * fmt1, val_data_np + 4 * fmt1))
    train_data_np = np.vstack((train_data_np, train_data_np + fmt2, train_data_np + 2 * fmt2, train_data_np + 3 * fmt2, train_data_np + 4 * fmt2))

    # 真实数据
    df1 = pd.DataFrame()
    df1["image_id"] = val_data_np[:, 0]
    df1["label"] = val_data_np[:, 1]
    df1.to_csv('./aug_data/'+str(i)+'_val5_data.csv')

    df2 = pd.DataFrame()
    df2["image_id"] = train_data_np[:, 0]
    df2["label"] = train_data_np[:, 1]

    df2.to_csv('./aug_data/'+str(i)+'_train5_data.csv')
    print("finish writting " + str(i)+" train5.csv")
print("finish writting train5.csv")

