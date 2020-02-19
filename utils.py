
import numpy as np
import pandas as pd
import os

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
    tt = pd.read_csv(data_path)
    tt = np.array(tt)
    
    x_train = tt[:, 1:]
    y_train = tt[:, 1]
    
    return x_train, y_train