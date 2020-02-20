import numpy as np
import torch
import os
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
noise_input = 256
lr = 0.0001

hidden = [512,256,256,256,512]

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
        test = torch.Tensor(np.random.normal(size = (100, noise_input))).to(device)
        temp = torch.sum(F.sigmoid(D(G(test))), dim = 0) / 100
        print(f"G loss : {G_loss}  | D loss  :  {D_loss}  | Time Spended : {time.time() - st}  |  Predict avg : {temp.data}")

        if epoch % 4 == 0:
            params = dict()
            params["G_weight"] = G.state_dict()
            params["D_weight"] = D.state_dict()
            torch.save(params, "./model.pt")



noise = np.random.normal(0, 1,size = (10, noise_input))
noise = torch.Tensor(noise).to(device)
noise.requires_grad_(False)
img = F.sigmoid(G(noise))

stack = []
for i in range(10):
    image = img[i]
    image = image.reshape(28,28).to("cpu") * 255
    image = np.array(image.detach(), dtype = np.uint8) 
    stack.append(image)

stack = np.concatenate(stack, axis = 1)
plt.imshow(stack)
plt.show()

#test = np.array(x_train[50] * 255, dtype= np.uint8).reshape(28,28)
#plt.imshow(test)