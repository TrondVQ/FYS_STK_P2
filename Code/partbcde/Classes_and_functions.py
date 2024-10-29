import numpy as np

# Functions:

def Design(x, p):
    """Create a design matrix for polynomial regression."""
    X = np.ones((x.shape[0], p + 1))
    for i in range(1, p + 1):
        X[:, i] = x[:, 0] ** i
    return X

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

# Defining some activation functions
def ReLU(z):
    return np.where(z > 0, z, 0)

# Derivative of the ReLU function
def ReLU_der(z):
    return np.where(z > 0, 1, 0)

# Leaky ReLU and its derivative
def leaky_ReLU(x, alpha=0.01):

    return np.where(x > 0, x, alpha * x)

def der_leaky_ReLU(x, alpha=0.01):
  
    return np.where(x > 0, 1, alpha)

def mean_squared_error(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def mean_squared_error_derivative(predictions, targets):
    return 2 * (predictions - targets) / targets.size

def linear(x):
    return x  # Linear activation

def linear_derivative(x):
    return np.ones_like(x)  # Derivative of linear is 1

def R2(z,zpred):
    return 1 - np.sum((z - zpred)**2) / np.sum((z - np.mean(z)) ** 2)

def CostLogReg(target):

    def func(X):
        
        return -(1.0 / target.shape[0]) * np.sum(
            (target * np.log(X + 10e-10)) + ((1 - target) * np.log(1 - X + 10e-10))
        )

    return func

# Log loss function:

def CostCrossEntropy(predictions, targets):
    return -(1.0 / targets.size) * np.sum(targets * np.log(predictions + 1e-10))

def CostCrossEntropyDer(predictions, targets):
    predictions = np.clip(predictions, 1e-10, 1 - 1e-10)  
    return -(targets / predictions) + (1 - targets) / (1 - predictions)

# Classes:

# Neural network class
class NetworkClass:
    def __init__(self, cost_fun, cost_der, network_input_size, layer_output_sizes, activation_funcs, activation_ders):
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.layers = []
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        # # Architecture check (activate for information about the network architecture)
        # print("Architecture configuration:")
        # print("--------------------------")
        # print("Network input size:", network_input_size)
        # print("Layer output sizes:", layer_output_sizes)
        # print("Activation functions:", activation_funcs)
        # print("Activation derivatives:", activation_ders)

        input_size = network_input_size
        for i, output_size in enumerate(layer_output_sizes):
            self.layers.append({
                'weights': np.random.randn(input_size, output_size) * np.sqrt(2. / input_size),
                'biases': np.zeros((1, output_size))
            })
            # print(f"Layer {i+1} weights shape:", self.layers[-1]['weights'].shape)
            # print(f"Layer {i+1} bias shape:", self.layers[-1]['biases'].shape)
            input_size = output_size
            


    def predict(self, inputs):
        activations = self._feed_forward_saver(inputs)
        return activations[-1]

    def cost(self, inputs, targets):
        predictions = self.predict(inputs)
        return self.cost_fun(predictions, targets)

    def _feed_forward_saver(self, inputs):
        activations = [inputs]
        for layer in self.layers:
            z = np.dot(activations[-1], layer['weights']) + layer['biases']
            a = self.activation_funcs[len(activations) - 1](z)
            activations.append(a)
        #print("activations:", activations)
        # print("activations input layer shape:", activations[0].shape)
        # print("activations hidden layer shape:", activations[-2].shape)
        # print("activations output shape:", activations[-1].shape)
        
        return activations

    def compute_gradient(self, inputs, targets):
        activations = self._feed_forward_saver(inputs)
        #print("activations shapes:", [a.shape for a in activations])
        layer_grads = []
        output = activations[-1]
        delta = self.cost_der(output, targets) * self.activation_ders[len(self.layers) - 1](output)

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            prev_activation = activations[i]
            layer_grads.append({
                'weights': np.dot(prev_activation.T, delta),
                'biases': np.sum(delta, axis=0, keepdims=True)
            })
            if i > 0:
                delta = np.dot(delta, layer['weights'].T) * self.activation_ders[i - 1](activations[i])

        layer_grads.reverse()
        return layer_grads

    def update_weights(self, layer_grads, learning_rate, lmbd):
        for i, layer in enumerate(self.layers):
            layer['weights'] -= learning_rate * (layer_grads[i]['weights'] + lmbd * layer['weights'])
            layer['biases'] -= learning_rate * layer_grads[i]['biases']



# Logistic regression class
class Logisticregr:
    def __init__(self, X_data, Y_data, eta=0.01, lmbd=0.0, iterations=100, nbatches=100, gamma=0.9):
        self.X_data = X_data
        self.Y_data = Y_data
        self.iterations = iterations
        self.eta = eta
        self.lmbd = lmbd
        self.nbatches = nbatches
        self.gamma = gamma
        self.beta_l = None

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    # Stochastic gradient descent from a) with some modifications. 
    def SGD(self, betainit=None):
        if betainit is None:
            betainit = np.random.rand(self.X_data.shape[1]) * 0.01  # Small random initialization

        beta = np.copy(betainit)
        betaold = np.zeros_like(beta)
        v = np.zeros_like(beta)

        print("Initial beta:", beta)

        ind = np.arange(self.X_data.shape[0])
        np.random.shuffle(ind)
        batch = np.array_split(ind, min(self.nbatches, len(ind)))

        for epoch in range(self.iterations):
         

            for k in range(len(batch)):
                betaold = np.copy(beta)

                xk = self.X_data[batch[k]]
                yk = self.Y_data[batch[k]]

                M = len(yk)

                predictions = self.sigmoid(xk @ beta)
                g = (1.0 / M) * xk.T @ (predictions - yk) + 2 * self.lmbd * beta
                #print("Gradient g:", g)  
                v = self.gamma * v + self.eta * g
                beta -= v

            #print("Updated beta:", beta)  # Debug print for beta updates

        print(f"Stopped after {self.iterations} epochs")
        self.beta_l = beta
        return beta

    def predict(self, X):
        return self.sigmoid(X @ self.beta_l)

    def predict_class(self, X, threshold=0.5):
        return self.predict(X) >= threshold

    def accuracy(self, X, y):
        y_pred = self.predict_class(X)
        return np.mean(y_pred == y)
    

