"""
FFNN with backpropagation

Discuss again your choice of cost function. ? In a it was CostOLS and CostRidge


Write an FFNN code for regression with a flexible number of hidden layers and nodes using the
Sigmoid function as activation function for the hidden layers.
 Initialize the weights using a normal distribution. How would you initialize the biases?
 And which activation function would you select for the final output layer?¨

 Train your network and compare the results with those from your OLS and Ridge Regression codes from project 1
  if you use the Franke function

Comment your results and give a critical discussion of the results obtained with the Linear Regression code and your own Neural Network code.
Make an analysis of the regularization parameters and the learning rates employed to find the optimal MSE and R2
 scores.
"""
import autograd.numpy as np  # We need to use this numpy wrapper to make automatic differentiation work later
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


#same as project 1
def FrankeFunction(x,y, noise = False):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    if noise:
        return term1 + term2 + term3 + term4  + 0.1 * np.random.randn(*x.shape)
    else:
        return term1 + term2 + term3 + term4


#From week 42
# Defining some activation functions
def ReLU(z):
    return np.where(z > 0, z, 0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=1)[:, np.newaxis]


def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time"""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)


def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(layer_output_size, i_size).T
        b = np.random.randn(layer_output_size)
        layers.append((W, b))

        i_size = layer_output_size
    return layers

inputs = np.random.rand(1000, 4) #hmm


def feed_forward_batch(inputs, layers, activation_funcs):
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = a @ W + b
        a = activation_func(z)
    return a

network_input_size = 4
layer_output_sizes = [12, 10, 3]
activation_funcs = [ReLU, ReLU, softmax]
layers = create_layers_batch(network_input_size, layer_output_sizes)

x = np.random.randn(network_input_size)
feed_forward_batch(inputs, layers, activation_funcs)