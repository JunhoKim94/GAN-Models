import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
import time
from model.model import MLP,GAN, DCGAN

print("\n ==============================> Training Start <=============================")
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")
print(torch.cuda.is_available())

x_train, y_train, x_test, y_test = load_mnist("./data/mnist")
#x_train, y_train, x_test, y_test =load_fashion_mnist("./data")

input_dim = x_train.shape[1]
total_data = x_train.shape[0]
print(input_dim)
classes = max(y_train)
batch = 500
epochs = 100
k = 2
noise_input = 100
lr = 0.0002
mode = None

#G 학습시키기가 어렵기 때무에 오히려 Shallow 한게 학습이 잘 됨
g_hidden = [256]
d_hidden = [256]

g_params = [(128, 7, 1, 0), (64, 4,2, 1), (1,4,2, 1)]
d_params = [(1,4,2, 1), (64,4,2, 1),(1,7,2, 0)]
#G = GAN(noise_input, 50, g_hidden, input_dim, classes, mode = mode).to(device)
#D = GAN(input_dim, 50, d_hidden, 1, classes, mode = mode).to(device)

G = DCGAN(100, g_params, mode = "g")
D = DCGAN(1, d_params, mode = "d")

G.to(device)
D.to(device)
'''
check = torch.load("./model.pt")
G.load_state_dict(check["G_weight"])
D.load_state_dict(check["D_weight"])
'''
criterion = nn.BCELoss()
G_optimizer = torch.optim.Adam(G.parameters(), lr = lr)
D_optimizer = torch.optim.Adam(D.parameters(), lr = lr)

st = time.time()
p_real = []
p_fake = []
for epoch in range(epochs + 1):
    
    for iteration in range(total_data // batch):

        for i in range(k):
            z = torch.randn(batch, noise_input).to(device)

            z = z.view(batch,noise_input,1,1)
            seed = np.random.choice(total_data, batch)
            x_batch = torch.Tensor(x_train[seed,:]).to(device)

            x_batch = x_batch.view(batch, 1, 28,28)
            y_class = torch.LongTensor(y_train[seed]).to(device)

            y_batch = torch.ones(batch).to(device)
            y_noise = torch.zeros(batch).to(device)

            G_out = G(z, y_class)
            D_out = D(x_batch, y_class)
            D_loss = criterion(D_out, y_batch) + criterion(D(G_out,y_class), y_noise)
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            
        z = torch.randn(batch, noise_input).to(device)
        
        z = z.view(batch, noise_input , 1, 1)
        #Minimize가 어려우니 Maximize를 하자 == label을 반대로 예측하자
        G_loss = criterion(D(G(z,y_class), y_class), y_batch)
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        #p_real.append(D_out)
        #p_fake.append(D(G_out, y_class))
        
    if epoch % 1 == 0:
        z = torch.randn(batch, noise_input).to(device)
        z = z.view(batch,noise_input,1,1)
        fake_label = torch.LongTensor(np.random.choice(10, size = (batch))).to(device)

        g_temp = torch.sum(D(G(z, fake_label),fake_label), dim = 0) / batch

        print(f"epoch : {epoch}  |  G loss : {G_loss}  | D loss  :  {D_loss}  | Time Spended : {time.time() - st}  |  Predict avg : {g_temp.data}")
        #이거 필수
        del G_out, D_out, D_loss, G_loss
        
        if epoch % 10 == 0:
            noise = torch.randn(20, noise_input).to(device)
            noise = noise.view(20, noise_input, 1, 1)
            fake_label = torch.LongTensor(np.array([0,1,2,3,4,5,6,7,8,9] * 2)).to(device)

            img = G(noise, fake_label)
            show(img, epoch)
            params = dict()
            params["G_weight"] = G.state_dict()
            params["D_weight"] = D.state_dict()
            torch.save(params, "./model.pt")
