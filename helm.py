#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import time
import scipy as sp
import os
import wget         # download file
import gzip         # gunzip file
import shutil       # save unziped file to disk
import scipy.io     # read .mat file


# ==================
# 1. Utils
# ==================

class MapMinMaxApplier(object):
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept
    def __call__(self, x):
        return x * self.slope + self.intercept
    def reverse(self, y):
        return (y-self.intercept) / self.slope

def mapminmax(x, ymin=-1, ymax=+1):
     x = np.asanyarray(x)
     xmax = x.max(axis=-1)
     xmin = x.min(axis=-1)
     if (xmax==xmin).any():
         raise ValueError("some rows have no variation")
     slope = ((ymax-ymin) / (xmax - xmin))[:,np.newaxis]
     intercept = (-xmin*(ymax-ymin)/(xmax-xmin))[:,np.newaxis] + ymin
     ps = MapMinMaxApplier(slope, intercept)
     return ps(x), ps

def download_ds(url_list):
    """
    Download and uncompress files.
    url_list:   list of urls
    """
    # Create destination folder
    if not os.path.exists('data'):
        os.mkdir('data')
    
    for url in url_list:
        fname = 'data/' + url.split('/')[-1]
        print('fname:', fname,'\nurl:', url)
        if not os.path.exists(fname[:-3]):
            wget.download(url, fname)
            # Uncompress file (gunzip)
            if fname[-3:] == '.gz':
                with gzip.open(fname, 'rb') as f_in:
                    with open(fname[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(fname)


# =========================================
# Load MNIST dataset
# =========================================
# Return  28x28x[number of MNIST images] matrix containing
# the raw MNIST images
def loadMNISTImages(filename):
    fp = open(filename, 'rb')
    
    assert fp != -1, 'Could not open ' + filename
    
    magic = np.fromfile(fp, dtype = '>u4', count = 1)
    assert magic == 2051, 'Bad magic number in ' + filename
    
    numImages = int(np.fromfile(fp, dtype = '>u4', count = 1))
    numRows = int(np.fromfile(fp, dtype = '>u4', count = 1))
    numCols = int(np.fromfile(fp, dtype = '>u4', count = 1))
    
    images = np.fromfile(fp, dtype = '>u1')
    images = np.reshape(images, (numCols, numRows, numImages), order = 'F')
    images = images.transpose(1, 0, 2)
    
    fp.close()
    
    # Reshape to #pixels x #examples
    images = np.reshape(images,
                        (images.shape[0] * images.shape[1], images.shape[2]),
                        order = 'F')
    # Convert to double and rescale to [0,1]
    images = images / 255
    
    return images


def loadMNISTLabels(filename):
    # loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
    # the labels for the MNIST images
    
    fp = open(filename, 'rb')
    assert fp != -1, 'Could not open ' + filename
    
    magic = np.fromfile(fp, dtype = '>u4', count = 1)
    assert magic == 2049, 'Bad magic number in ' + filename
    
    numLabels = np.fromfile(fp, dtype = '>u4', count = 1)
    labels = np.fromfile(fp, dtype = '>u1')
    labels = labels.reshape(len(labels), 1, order = 'F')
    
    assert labels.shape[0] == numLabels, 'Mismatch in label count'
    
    fp.close()
    y = np.zeros((len(labels), 10))
    for i in range(10):
        y[:,i:i+1] = (labels == i) * 1
        
    y = np.where(y == 0, -1, y)
    
    return y

def norb_url():
    url = ['https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x01235x9x18x6x2x108x108-testing-01-cat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x01235x9x18x6x2x108x108-testing-01-dat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x01235x9x18x6x2x108x108-testing-01-info.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x01235x9x18x6x2x108x108-testing-02-cat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x01235x9x18x6x2x108x108-testing-02-dat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x01235x9x18x6x2x108x108-testing-02-info.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-01-cat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-01-dat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-01-info.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-02-cat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-02-dat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-02-info.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-03-cat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-03-dat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-03-info.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-04-cat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-04-dat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-04-info.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-05-cat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-05-dat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-05-info.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-06-cat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-06-dat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-06-info.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-07-cat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-07-dat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-07-info.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-08-cat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-08-dat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-08-info.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-09-cat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-09-dat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-09-info.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-10-cat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-10-dat.mat.gz',
            'https://cs.nyu.edu/~ylclab/data/norb-v1.0/norb-5x46789x9x18x6x2x108x108-training-10-info.mat.gz']
    return url

# ==================
# Demo MNIST
# ==================
def demo_mnist():
    """
    Example of H-ELM with MNIST dataset.
    """
    # Dataset: http://yann.lecun.com/exdb/mnist/
   
    ## This is the demo for MNIST dataset.
    np.random.rand('state', 0)
    
    #-----------------------------
    # Prepare dataset
    #-----------------------------
    # Dataset
    # We used the original MNIST dataset, the overall samples are simply scaled
    # to [0, 1].
    # And the labels are resized to 10 dimension which are -1 or 1.
    
    url = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
           'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
           'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
           'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
    
    # Download MNIST files
    download_ds(url)
    X_train = loadMNISTImages('data/train-images-idx3-ubyte')
    y_train = loadMNISTLabels('data/train-labels-idx1-ubyte')
    X_test  = loadMNISTImages('data/t10k-images-idx3-ubyte')
    y_test  = loadMNISTLabels('data/t10k-labels-idx1-ubyte')

    # Use small portion of dataset (1000 samples)
    N = 1000
    X_train = X_train[:N]
    y_train = y_train[:N]
    X_test  = X_test[:N]
    y_test  = y_test[:N]
    
    ## Randomness
    # Due to the random initialization of neural networks, it is very common
    # to see that the performance has fluctuations.
    # To reproduce the exact same testing results in our paper, please use the
    # attached .mat file obtained by following codes:
    
    # TODO: define N1, N2 and N3
    np.random.seed(16917921)   # 5000
    np.random.seed(67797325)   # 12000
    b1 = 2 * np.random.rand(X_train.shape[1] + 1, N1) - 1
    b2 = 2 * np.random.rand(N1 + 1, N2) - 1
    b3 = sp.linalg.orth(2 * np.random.rand(N2 + 1, N3).T - 1).T
    
    # And our testing hardware and software conditions are as follows:
    # Laptop, Intel-i7 2.4G CPU, 16G DDR3 RAM, Windows 7, Matlab R2013b.
    # (We have also tested the codes on Matlab 2014b)
    
    # If you have further interests, you could try to build random mapping
    # matrices, add other preprocessing or tuning tricks by you own, :-).
    ## To achieve the 99.13% accuracy of H-ELM as we did in paper, load the
    # random matrices which are totally independent from the training data.
    C = 2 ^ -30
    s = .8  # C is the L2 penalty of the last layer ELM and s is the scaling factor.
    #load random_700_700_12000.mat
    
    ## If the RAM of your computer is less than 16G, you may try the following
    # version which has less than half of the hidden nodes
    # C = 2^-30
    # s = 1
    # load random_700_700_5000.mat
    
    ## Call the training function
    TrainingAccuracy, TestingAccuracy, Training_time, Testing_time = helm_train(
        X_train, y_train, X_test, y_test, b1, b2, b3, s, C)

    print('Training accuracy:', TrainingAccuracy)
    print('Testing accuracy:', TestingAccuracy)
    print('Training time:', Training_time)
    print('Testing time:', Testing_time)

    return


# ==================
# Demo NORB
# ==================
def demo_norb():
    # Dataset: https://cs.nyu.edu/~ylclab/data/norb-v1.0/
    
    ## This is the demo for NORB dataset.
    # Donwload dataset
    url = norb_url()
    download_ds(url)
    # Read dataset from file
    for fname in os.listdir('data/*.mat'):
        mat = scipy.io.loadmat('data/fname')
        
    # TODO: Create train and test sets
    # TODO: Define N1, N2 and N3
    # Dataset
    # We used the original NORB dataset, the overall samples are simply scaled
    # to [0,1].
    # And the labels are resized to 5 dimensions -1 or 1.
    #load norb.mat
    X_train = X_train.reshape(2048, 24300, order = 'F').copy().T
    X_test  = X_test.reshape(2048, 24300, order = 'F').copy().T
    
    ## zscore and ZCA whiten
    train_x = (X_train - np.mean(X_train, 2)) / np.sqrt(np.var(X_train, 1) + 10)
    
    C = np.cov(X_train)
    M = np.mean(X_train)
    D, V = np.linalg.eig(C)
    P = V @ np.diag(np.sqrt(1. / (np.diag(D) + 1e2))) @ V.T
    X_train = (X_train - M) @ P
    X_test  = (X_test - np.mean(X_test, 1)) / np.sqrt(np.var(X_test, 2) + 10)
    test_x = (X_test - M) @ P
    
    ## Randomness
    # Due to the random initialization of neural networks, it is very common
    # to see that the performance has fluctuations.
    # To reproduce the exact same testing results in our paper, please use the
    # attached .mat file obtained by following codes:
    
    # np.random.rand('state', 78309924)
    b1 = 2 * np.random.rand(X_train.shape[1] + 1, N1) - 1
    b2 = 2 * np.random.rand(N1 + 1, N2) - 1
    b3 = sp.linalg.orth(2 * np.random.rand(N2 + 1, N3).T - 1).T
    
    # And our testing hardware and software conditions are as follows:
    # Laptop, Intel-i7 2.4G CPU, 16G DDR3 RAM, Windows 7, Matlab R2013b.
    # (We have also tested the codes on Matlab 2014b)
    
    # If you have further interests, you could try to build random mapping
    # matrices, add other preprocessing or tuning tricks by you own, :-).
    ## To achieve the 91.28% H-ELM we as we did in paper, load the random
    # matrices which are totally independent from the training data.
    C = 2 ^ -30;
    s = .8
    #load random_3000_3000_15000.mat
    ##
    TrainingAccuracy, TestingAccuracy, Training_time, Testing_time = helm_train(
        X_train, y_train, X_test, y_test, b1, b2, b3, s, C)
    
    print('Training accuracy:', TrainingAccuracy)
    print('Testing accuracy:', TestingAccuracy)
    print('Training time:', Training_time)
    print('Testing time:', Testing_time)
    return


# ==================
# Train
# ==================
def helm_train(train_x, train_y, test_x, test_y, b1=None, b2=None, b3=None, s=0, C=0):
    start = time.time()
    train_x = sp.zscore(train_x.T, axis = 0, ddof=0).T
    H1 = np.hstack(train_x, 0.1 * np.ones((len(train_x), 1)))
    del(train_x)
    
    ## First layer RELM
    A1 = H1 @ b1
    A1 = mapminmax(A1)
    del(b1)
    beta1 = sparse_elm_autoencoder(A1, H1, 1e-3, 50).T
    del(A1)
    
    T1 = H1 @ beta1
    print('Layer 1: Max Val of Output %f Min Val %f', np.max(T1), np.min(T1))
    
    T1, ps1 = mapminmax(T1.T, 0, 1)
    T1 = T1.T
    
    del(H1)
    ## Second layer RELM
    H2 = np.hstack(T1, 0.1 * np.ones((len(T1), 1)))
    del(T1)
    
    A2 = H2 @ b2
    A2 = mapminmax(A2)
    del(b2)
    beta2 = sparse_elm_autoencoder(A2, H2, 1e-3, 50).T
    del(A2)
    
    T2 = H2 @ beta2
    print('Layer 2: Max Val of Output %f Min Val %f', np.max(T2), np.min(T2))
    
    T2, ps2 = mapminmax(T2.T, 0, 1)
    T2 = T2.T
    del(H2)
    
    ## Original ELM
    H3 = np.hstack(T2, 0.1 * np.ones((len(T2), 1)))
    del(T2)
    
    T3 = H3 @ b3
    l3 = np.max(T3)
    l3 = s / l3
    print('Layer 3: Max Val of Output %f Min Val %f', l3, np.min(T3))
    
    T3 = np.tanh(T3 @ l3)
    del(H3)
    
    ## Finsh Training
    a = (T3.T @ T3 + np.eye(len(T3.T)) @ C)
    if a.shape[0] == a.shape[1]:
        beta = np.linalg.solve(a, (T3.T @ train_y))
    else:
        beta = np.linalg.lstsq(a, (T3.T @ train_y))
    Training_time = start - time.time()
    print('Training has been finished!')
    print('The Total Training Time is: ', str(Training_time), ' seconds')
    
    ## Calculate the training accuracy
    xx = T3 @ beta
    del(T3)
    
    yy = result_tra(xx)
    train_yy = result_tra(train_y)
    TrainingAccuracy = len(np.sum(yy == train_yy)) / len(train_yy)
    print('Training Accuracy is:', str(TrainingAccuracy * 100), '%')
    
    ## First layer feedforward
    start = time.time()
    
    test_x = sp.zscore(test_x.T, axis = 0, ddof=0).T
    HH1 = np.hstack(test_x, 0.1 * np.ones((len(test_x), 1)))
    del(test_x)
    
    TT1 = HH1 @ beta1
    TT1 = mapminmax('apply', TT1.T, ps1).T
    del(HH1)
    del(beta1)
    
    ## Second layer feedforward
    HH2 = np.hstack(TT1, 0.1 * np.ones((len(TT1), 1)))
    del(TT1)
    
    TT2 = HH2 @ beta2
    TT2 = mapminmax('apply', TT2.T, ps2).T
    
    del(HH2)
    del(beta2)
    
    ## Last layer feedforward
    HH3 = np.stack(TT2, 0.1 * np.ones((len(TT2), 1)))
    del(TT2)
    
    TT3 = np.tanh(HH3 @ b3 @ l3)
    del(HH3)
    del(b3)
    
    x = TT3 @ beta
    y = result_tra(x)
    test_yy = result_tra(test_y)
    TestingAccuracy = len(np.sum(y == test_yy)) / len(test_yy)
    del(TT3)
    
    ## Calculate the testing accuracy
    Testing_time = time.time()
    print('Testing has been finished!')
    print('The Total Testing Time is:', str(Testing_time), ' seconds')
    print('Testing Accuracy is:', str(TestingAccuracy * 100), ' %')
    
    return TrainingAccuracy, TestingAccuracy, Training_time, Testing_time


# ==================
# Result_tra
# ==================
def result_tra(x):

    y = np.zeros(len(x))
    for i in range(len(x)):
        _, y[i] = np.max(x[i,:])

    return y.T

# ==================
# Sparse ELM autoencoder
# ==================
def sparse_elm_autoencoder(A, b, lam, itrs):

    AA = (A.T) @ A
    
    Lf = np.max(np.linalg.eigvals(AA))
    Li = 1 / Lf
    alp = lam @ Li
    m = np.shape(A, 1)
    n = np.shape(b, 1)
    x = np.zeros((m, n))
    yk = x
    tk = 1
    L1 = 2 * Li @ AA
    L2 = 2 * Li @ A.T @ b
    for i in range(itrs):
        ck = yk - L1 @ yk + L2
        x1 = (max(abs(ck) - alp, 0)) * np.sign(ck)
        tk1 = 0.5 + 0.5 * np.sqrt(1 + 4 * tk ** 2)
        tt = (tk - 1) / tk1
        yk = x1 + tt @ (x - x1)
        tk = tk1
        x = x1

    return x


