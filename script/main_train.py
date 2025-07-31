__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/12/17 16:45:38"

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import *
from prep_data import *
import sys
import os

#### preparing dataset
# with open("./data/MNIST.pkl", 'rb') as file_handle:
#     MNIST = pickle.load(file_handle)

# data = MNIST_Dataset(MNIST['train_image'])

ROOT_DIR = "/mnt/TD-VAE"

#Load MNIST data (train + test)
from torchvision.datasets import MNIST
import numpy as np

# Load train and test datasets
train_dataset = MNIST(root=f"{ROOT_DIR}/data", train=True, download=True)
train_images = np.stack([np.array(img, dtype=np.uint8) for img, _ in train_dataset])[:1000] #TODO CHANGED TO 1000  # (60000, 28, 28)
data = MNIST_Dataset(train_images)


batch_size = 512
data_loader = DataLoader(data,
                        batch_size = batch_size,
                        shuffle = True)

#### build a TD-VAE model
input_size = 784
processed_x_size = 784
belief_state_size = 50
state_size = 8
tdvae = TD_VAE(input_size, processed_x_size, belief_state_size, state_size)
tdvae = tdvae.cuda()

#### training
optimizer = optim.Adam(tdvae.parameters(), lr = 0.0005)
num_epoch = 4000
log_file_handle = open(f"{ROOT_DIR}/log/loginfo_new.txt", 'w')
save_dir = f"{ROOT_DIR}/output/model_with_kl"
os.makedirs(save_dir, exist_ok=True)
for epoch in range(num_epoch):
    for idx, images in enumerate(data_loader):        
        images = images.cuda()       
        tdvae.forward(images)
        t_1 = np.random.choice(16)
        t_2 = t_1 + np.random.choice([1,2,3,4])
        loss, kl_loss = tdvae.calculate_loss(t_1, t_2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}".format(epoch, idx, loss.item()),
            file = log_file_handle, flush = True)
        
        # print(kl_loss, type(kl_loss))
        # print(kl_loss.item())
        print("epoch: {:>4d}, idx: {:>4d}, loss: {:.2f}, kl_loss: {:.2f}".format(epoch, idx, loss.item(), kl_loss.item()))

    # if (epoch + 1) % 50 == 0:
    torch.save({
        'epoch': epoch,
        # 'model_state_dict': tdvae.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'kl_loss': kl_loss
    }, f"{save_dir}/new_model_epoch_{epoch}.pt")
