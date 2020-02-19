import numpy as np
import torch
import os
import cv2
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
import time
from model.model import MLP


print("\n ==============================> Training Start <=============================")
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")
print(torch.cuda.is_available())

x_train, y_train = load_mnist("./data/mnist/mnist_train.csv")
#x_train, y_train, x_test, y_test =load_fashion_mnist("./data")

input_dim = x_train.shape[1]
total_data = x_train.shape[0]
batch = 500
classes = max(y_train)
epochs = 10
k = 3
noise_input = 128
lr = 0.001

hidden = [512,256,128,256,512]

G = MLP(noise_input, hidden, input_dim).to(device)
D = MLP(input_dim, hidden, 1).to(device)

criterion = nn.BCEWithLogitsLoss()
G_optimizer = torch.optim.Adam(G.parameters(), lr = lr)
D_optimizer = torch.optim.Adam(D.parameters(), lr = lr)
st = time.time()
for epoch in range(epochs + 1):
    
    for iteration in range(total_data // batch):

        for i in range(k):
            z = np.random.normal(size = (batch, noise_input))
            z = torch.Tensor(z).to(device)

            seed = np.random.choice(total_data, batch)
            x_batch = torch.Tensor(x_train[seed,:]).to(device)
            y_batch = torch.ones(batch,1).to(device)
            y_noise = torch.zeros(batch,1).to(device)

            G_out = G(z)
            D_loss = criterion(D(x_batch), y_batch) + criterion(D(G(z)), y_noise)
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

        G_loss = criterion(D(G(z)), y_batch)
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        
    if epoch % 1 == 0:
        
        print(f"G loss : {G_loss}  | D loss  :  {D_loss}  | Time Spended : {time.time() - st}")

noise = np.random.normal(size = (10, noise_input))
noise = torch.Tensor(noise).to(device)
noise.requires_grad_(False)
img = G(noise)

i = np.random.choice(10,1)
image = img[i]
image = image.reshape(28,28).to("cpu")
image = np.array(image.detach(), dtype = np.uint8)
plt.imshow(image)
plt.show()

#test = np.array(x_train[50] * 255, dtype= np.uint8).reshape(28,28)
#plt.imshow(test)