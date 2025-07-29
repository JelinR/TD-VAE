import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from model import *
from prep_data import *

""" After training the model, we can try to use the model to do
jumpy predictions.
"""
ROOT_DIR = "/mnt/TD-VAE"
EPOCH_NUM = 350

#### load trained model

checkpoint = torch.load(f"{ROOT_DIR}/output/model/new_model_epoch_{EPOCH_NUM}.pt")
input_size = 784
processed_x_size = 784
belief_state_size = 50
state_size = 8
tdvae = TD_VAE(input_size, processed_x_size, belief_state_size, state_size)
optimizer = optim.Adam(tdvae.parameters(), lr = 0.0005)

tdvae.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#### load dataset 
# with open("./data/MNIST.pkl", 'rb') as file_handle:
#     MNIST = pickle.load(file_handle)

# data = MNIST_Dataset(MNIST['train_image'], binary = False)



#Load MNIST data (train + test)
from torchvision.datasets import MNIST
import numpy as np

# Load train and test datasets
train_dataset = MNIST(root=f"{ROOT_DIR}/data", train=True, download=True)
train_images = np.stack([np.array(img, dtype=np.uint8) for img, _ in train_dataset])  # (60000, 28, 28)
data = MNIST_Dataset(train_images, binary=False)



tdvae.eval()
tdvae = tdvae.cuda()


batch_size = 6
data_loader = DataLoader(data,
                         batch_size = batch_size,
                         shuffle = True)
idx, images = next(enumerate(data_loader))

images = images.cuda()

## calculate belief
tdvae.forward(images)

## jumpy rollout
t1, t2 = 11, 15
rollout_images = tdvae.rollout(images, t1, t2)

#### plot results
#fig = plt.figure(0, figsize = (t2+2,batch_size))
fig = plt.figure(0, figsize = (12,4))

#fig = plt.figure(0)
fig.clf()
gs = gridspec.GridSpec(batch_size,t2+2)
gs.update(wspace = 0.05, hspace = 0.05)
for i in range(batch_size):
    for j in range(t1):
        axes = plt.subplot(gs[i,j])
        axes.imshow(1-images.cpu().data.numpy()[i,j].reshape(28,28),
                    cmap = 'binary')
        axes.axis('off')

    for j in range(t1,t2+1):
        axes = plt.subplot(gs[i,j+1])
        axes.imshow(1-rollout_images.cpu().data.numpy()[i,j-t1].reshape(28,28),
                    cmap = 'binary')
        axes.axis('off')

fig.savefig(f"{ROOT_DIR}/output/rollout_result_{EPOCH_NUM}.eps")
fig.savefig(f"{ROOT_DIR}/output/rollout_result_{EPOCH_NUM}.png")
plt.show()
sys.exit()
