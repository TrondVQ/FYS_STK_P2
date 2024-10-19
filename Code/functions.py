import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from random import random, seed

def Design(x,degree):
    X = np.zeros((len(x[:,0]),degree+1))
    for i in range(degree+1):
        X[:,i] = x[:,0]**i
    return X


def OLS(X, z):
    beta = np.linalg.pinv(X.T @ X) @ X.T @ z
    return beta


def Ridge(X,z,lam):
    beta = np.linalg.pinv(X.T @ X + lam * np.eye(X.shape[1])) @ X.T @ z
    return beta


def MSE(z,zpred): #z=target data, zpred=predicted target data
    return np.mean((z-zpred)**2)


def R2(z,zpred):
    return 1 - np.sum((z - zpred)**2) / np.sum((z - np.mean(z)) ** 2)
