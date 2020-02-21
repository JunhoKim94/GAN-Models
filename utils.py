
import numpy as np
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import cv2

def load_fashion_mnist(data_path):
    cifar_path = os.path.join(data_path, 'fashion_mnist')

    x_train = np.load(os.path.join(cifar_path, 'fashion_train_x.npy'))
    y_train = np.load(os.path.join(cifar_path, 'fashion_train_y.npy'))
    x_test = np.load(os.path.join(cifar_path, 'fashion_test_x.npy'))
    y_test = np.load(os.path.join(cifar_path, 'fashion_test_y.npy'))

    # Flatten X
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)

    return x_train, y_train, x_test, y_test

def load_mnist(data_path):
    train_path = data_path + "/mnist_train.csv"
    test_path = data_path + "/mnist_test.csv"
    train = np.array(pd.read_csv(train_path))
    test = np.array(pd.read_csv(test_path))
    
    seed = np.random.permutation(len(train))

    x_train = train[seed, 1:] / 255
    #classes =max(tt[:,1])
    y_train = train[seed, 0]
    x_test = test[:,1:] /255
    y_test = test[:,0]

    return x_train, y_train, x_test, y_test

def show(img, epoch, save = True):

    img_show = []
    for j in range(2):
        stack = []
        for i in range(10):
            image = img[10*j + i]
            image = image.reshape(28,28).to("cpu") * 255
            image = np.array(image.detach(), dtype = np.uint8) 
            stack.append(image)

        stack = np.concatenate(stack, axis = 1) 
        img_show.append(stack)
    img_show = np.concatenate(img_show, axis = 0)
    
    img_show = cv2.resize(img_show, (800, 200), interpolation= cv2.INTER_LINEAR)
    cv2.imshow("test",img_show)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    
    if save:
        cv2.imwrite("./trunk/epoch_%d.jpg"%epoch, img_show)

