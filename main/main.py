#!/usr/bin/env mlp
# -*- coding: utf-8 -*-
""":cvar

"""
# Import standard libraries
import numpy as np
import pandas as pd
import timeit
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
# Import custom libraries
from ELM.elm import ELM
import neuralnets as nn


# ======================
# Data normalization
# ======================
# Each dependent variable is normalizaded using min-max or z-score.
# min-max transforms the values to fit in the range [0, 1].
# The formula is:
# x = (x - xmin) / (xmax - xmin),
# where 'xmax' is the maximum value and 'xmin' is the minimum value
# z-score uses normal distribution with mean = 0 and standard deviation = 1
def normalize(X, method, mu = 0, sd = 0, xmin = 0, xmax = 0):
    # Min - max
    if method == 'min-max':
        if xmin == 0 or xmax == 0:
            xmin = np.min(X)
            xmax = np.max(X)
        return (X - xmin) / (xmax - xmin), xmin, xmax
    
    # Norm Z
    elif method == 'z-score':
        if mu == 0 or sd == 0:
            mu = X.mean()
            sd = np.std(X)
        return (X - mu) / sd, mu, sd
    
    return None

def norm(X_train, X_test):

    for i in range(X_train.shape[1]):
        X_train[:, i], xmin, xmax = normalize(X = X_train[:, i],
                                              method = 'min-max')
        X_test[:, i], xmin, xmax = normalize(X = X_test[:, i], xmin = xmin,
                                             xmax = xmax, method = 'min-max')


# ==================================
# Load datasets
# - download the dataset
# - convert to one-hot encoding
# - create training and test sets
# - normalize X values in the range [0, 1]
# ==================================
# Multiple classes datasets
# ==================================

def load_dataset(dataset):
    dsname, ver = dataset  # dataset name and version
    if dsname == 'breast-cancer':
        ds = datasets.load_breast_cancer()
    else:
        ds = fetch_openml(name = dsname, version = ver)
    
    n_class = len(np.unique(ds.target))
    
    # Convert to one-hot encoded
    y = np.zeros((len(ds.target), n_class))
    for i in range(n_class):
        y[:, i] = np.where(ds.target == np.unique(ds.target)[i], 1, 0)
    
    # Normalize data
    X, _, _ = normalize(ds.data, method = 'z-score')

    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    return X_train, y_train, X_test, y_test

#========================================
# Main
#
# ==================================
#  Train and Predict ML Models
# - Extreme Learning Machine (ELM)
# - Single Layer Perceptron (SLP)
# - Multi Layer Perceptron (MLP)
# - Radial-Basis-Function Network (RBF)
# ==================================

if __name__ == '__main__':
    
    K = 5
    train_acc = np.zeros(K)
    train_err = np.zeros(K)
    test_acc = np.zeros(K)
    test_err = np.zeros(K)
    
    # Result table
    result = pd.DataFrame(
        columns = ['Dataset', 'Model', 'Number of Neurons',
                   'Train Accuracy (%)', 'Validation Accuracy (%)',
                   'Test Accuracy (%)', 'Velocity (s)',
                   'Standard Error', 'CI Lower Bound', 'CI Upper Bound'])
    
    # Include new datasets in this list
    # list of tuples: (dataset name, dataset version)
    dataset = [('iris', 1), ('wine', 1), ('dna', 1), ('breast-cancer', 1),
               ('Australian', 4), ('diabetes', 1)]
    
    models = {'ELM': {'train': nn.elm_train, 'predict': nn.elm_predict},
              'SLP': {'train': nn.slp_train, 'predict': nn.slp_predict},
              'MLP': {'train': nn.mlp_train, 'predict': nn.mlp_predict},
              'RBF': {'train': nn.rbf_train, 'predict': nn.rbf_predict}
              }

    for ds in dataset:
        print('Processing:', ds[0])
    
        # Load dataset
        X_train, y_train, X_test, y_test = load_dataset(dataset = ds)
    
        # Execute all models
        for model in models:
            print('Model: {}'. format(model))
            
            max_val_acc = 0
            Lmax=0
            #-----------------------
            # Use holdout to find the best number of hidden layer
            # neurons for each model.
            # -----------------------
#            for L in [10, 50, 100, 200, 300]:
            for L in [100]:
                # Train data
                # elm.train(X_train, y_train.reshape(-1, 1))
                params = models[model]['train'](X_train, y_train,
                                                input_size = X_train.shape[1],
                                                hidden_size = len(X_test),
                                                L = L,      # Number of hidden layer neurons
                                                maxiter = 100,  # Maximum number of epochs
                                                plot = False,   # create plot
                                                pdir = '', DS_name = '')
                
                # Validation Accuracy
                y_hat = models[model]['predict'](X_test, params)
                y_hat = (y_hat == np.max(y_hat, axis = 1, keepdims = True)) * 1
                val_acc = np.round(100 * np.mean(y_test == y_hat), 2)
                
                # Keep parameters of best validation accuracy
                if val_acc > max_val_acc:
                    max_val_acc = val_acc   # accuracy
                    Lmax = L            # number of hidden layer neurons
                    e_hat = 1 - np.mean(y_test == y_hat)         # validation error
                    se = np.sqrt(e_hat * (1-e_hat)/len(X_test))  # standard error

                # Exit cross validation if model is SLP, since it has no hidden layer neurons
                if model in ['SLP', 'RBF']:
                    break

            #-----------------------
            # k-fold cross validation to compute true error and confidence interval
            #-----------------------
            N = len(X_train)
            for k in range(K):
                # Create train and validation sets
                # Define K% of rows for validation set
                rv = np.array(range(int(N * k / K),
                                    int(N * (k + 1) / K)))

                # Define complementary row numbers for train set
                r = np.setdiff1d(np.array(range(X_train.shape[0])), rv)
                X1, y1 = X_train[r], y_train[r]     # Train set
                X2, y2 = X_train[rv], y_train[rv]   # Validation set
                
                # Train and test with best parameters
                start_time = timeit.default_timer()
                params = models[model]['train'](X1, y1,
                                                input_size = X1.shape[1],
                                                hidden_size = len(X2),
                                                L = Lmax,  # Number of hidden layer neurons
                                                maxiter = 100, # Maximum number of epochs
                                                plot = False,  # create plot
                                                pdir = '', DS_name = '')

                # Train accuracy and loss
                y_hat = models[model]['predict'](X1, params)
                y_hat = (y_hat == np.max(y_hat, axis = 1, keepdims = True)) * 1
                train_acc[k] = np.round(100 * np.mean(y1 == y_hat), 2)
                train_err[k] = 1 - np.mean(y1 == y_hat)
                
                # Test accuracy and loss
                y_hat = models[model]['predict'](X2, params)
                y_hat = (y_hat == np.max(y_hat, axis = 1, keepdims = True)) * 1
                test_acc[k] = np.round(100 * np.mean(y2 == y_hat), 2)
                test_err[k] = 1 - np.mean(y2 == y_hat)

            # Cross-validation accuracy and loss
            cv_train_acc = np.mean(train_acc)
            cv_train_err = np.mean(train_err)
            cv_test_acc = np.mean(test_acc)
            cv_test_err = np.mean(test_err)

            # Standard error and Confidence Interval
            se = np.sqrt(cv_train_err * (1-cv_train_err) / N)   # standard error
            cilb = cv_train_err - 1.96 * se     # confidence interval - lower bound
            ciub = cv_train_err + 1.96 * se     # confidence interval - upper bound
            
            # Update result table
            end_time = timeit.default_timer()
            velocity = end_time - start_time
            result.loc[len(result) + 1] = [ds[0].title(), model, L,
                                           cv_train_acc, max_val_acc, cv_test_acc,
                                           velocity, se, cilb, ciub]

    # Save and print result table
    result.to_csv("result.csv",  sep='\t', index=False)
    print(result)

    #========================================
    # Create scatter plots
    #========================================
    model = [model for model in np.unique(result['Model'])]
    for ds in np.unique(result['Dataset']):
   
        x = result[result['Dataset'] == ds]['Number of Neurons']
        y1 = result[result['Dataset'] == ds]['Train Accuracy (%)']
        y2 = result[result['Dataset'] == ds]['Test Accuracy (%)']

        _, (ax1, ax2) = plt.subplots(1, 2)
        for i in range(len(model)):
            ax1.scatter(x.iloc[i], y1.iloc[i], s = 80, label = model[i])  # Train
            ax2.scatter(x.iloc[i], y2.iloc[i], s = 80, label = model[i])  # Test

        # Train accuracy
        ax1.set_title(ds + ' - Train Accuracy (%)')
        ax1.set_xlabel('Number of Neurons')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()

        # Test accuracy
        ax2.set_title(ds + ' - Test Accuracy (%)')
        ax2.set_xlabel('Number of Neurons')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()

        # Save and show plot
        plt.savefig('../plots/' + ds + '.png',
                    dpi = 300, bbox_inches = 'tight')
        plt.show()

    print(1)
# TODO - verificar o resultado do DNA com MLP
# - embaralhar as amostras e usar cross validation p/ testar
# - escolher melhor numero de neuronios para cada modelo
# - fazer holdout p/ escolher o melhor parametro
# - dividir cj de dados: 10% p/ validação p/ escolher parametros do modelo
#
# ex: 200 instacias: 10% p/ validação (20 instancias)
# Passo 0: separar cj de treino, validação e teste
# Passo 1: treinar modelo com 10, 20, 30... nós na camada intermediaria e testar na validação para escolher melhor modelo
# Passo 2: fazer cross validation com treino + validação e calcular erro verdadeiro e intervalo de confiança
# passo 3: usar melhor valor no teste e comparar erro com passo 2
