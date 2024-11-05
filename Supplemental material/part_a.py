"""
A plain gradient descent with a fixed learning rate (you will need to tune it) using the analytical expression for the gradient.

Add momentum to the plain GD code and compare convergence with a fixed learning rate (you may need to tune the learning rate). Keep using the analytical expression for the gradient.

Repeat these steps for stochastic gradient descent with mini batches and a given number of epochs. Use a tunable learning rate as discussed in the lectures from weeks 39 and 40. Discuss the results as functions of the various parameters (size of batches, number of epochs etc). Use the analytical gradient.

Implement the Adagrad method in order to tune the learning rate. Do this with and without momentum for plain gradient descent and SGD.

Add RMSprop and Adam to your library of methods for tuning the learning rate.

Then compare again with : -> Replace thereafter your analytical gradient with either Autograd or JAX
"""

#KUN DRAFT -> BRUKES IKKE
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler

def Simple_design_matrix(x,degree):
    X = np.zeros((len(x[:,0]),degree+1))
    for i in range(degree+1):
        X[:,i] = x[:,0]**i
    return X


# Cost functions>  Taken from week 39: Using Autograd with OLS
def CostOLS(X, y, beta, n): #basiccly mse
    return (1.0/n) * np.sum((y - X @ beta)**2)

def CostRidge(X, y, beta, lmbda, n):
    return (1.0/n) * np.sum((y - X @ beta)**2) + lmbda * np.sum(beta**2)

# Gradient Descent
def GD (X, y, beta, n = 100, eta = 0.001, ridge = False, momentum = False,  delta_momentum = 0.3, change = 0.0, lmbda = 0.01, Niterations = 100):
    cost = []
    n_samples = len(y)
    #Momemtum code taken from Week 39: Same code but now with momentum gradient descent
    for iter in range(Niterations):
        if ridge == False: #lin reg
            gradient = (2.0/n)*X.T @ (X @ beta-y) # kin
            cost.append(CostOLS(beta))
        elif ridge == True:
            # Taken from week 39: Program example for gradient descent with Ridge Regression
            gradient = (2.0 / n) * X.T @ (X @ (beta)-y) + 2 * lmbda * beta
            cost.append(CostRidge(beta, ridge))
        if momentum:
            new_change = eta * gradient + delta_momentum * change
            beta -= new_change
            change = new_change
        else:
            beta -= eta * gradient

    return beta, cost


#Code taken and modified from week 39: Code with a Number of Minibatches which varies, analytical gradient
def learning_schedule(t, t0=5, t1=50):
    return t0/(t+t1)

#Code taken and modified from week 39: Code with a Number of Minibatches which varies, analytical gradient
#SGD with momentum and ridge + Fiks
def SGD(X, y, theta, n = 100, n_epochs=50, batch_size=5, momentum=0.8, change=0.1, ridge=False, lmbda=0.01, t0=5, t1=50):
    M = batch_size
    m = int(n / M)  # Number of mini-batches
    cost = []

    for epoch in range(n_epochs):
        for i in range(m):
            random_index = M * np.random.randint(m)
            xi = X[random_index:random_index + M]
            yi = y[random_index:random_index + M]

            if ridge:
                gradient = (2.0 / M) * xi.T @ (xi @ theta - yi) + 2 * lmbda * theta
                cost.append(CostRidge(theta, lmbda))
            else:
                gradient = (2.0 / M) * xi.T @ (xi @ theta - yi)
                cost.append(CostOLS(theta))

            eta = learning_schedule(epoch * m + i, t0, t1)
            new_change = eta * gradient + momentum * change
            theta -= new_change
            change = new_change

    return theta, cost


def AdagradGD(X, y, beta, n = 100, eta = 0.001, lmbda = 0.01, Niterations = 100, ridge = False, momentum = False,  delta_momentum = 0.3, change = 0.0, Giter = 0.0, delta=1e-8):
    cost = []
    for iter in range(Niterations):
        if ridge == False:
            gradient = (2.0 / n) * X.T @ (X @ beta - y)
            cost.append(CostOLS(beta))
        else:
            gradient = (2.0 / n) * X.T @ (X @ beta - y) + 2 * lmbda * beta
            cost.append(CostRidge(beta, lmbda))

        Giter += gradient**2
        update = eta * gradient / (delta + np.sqrt(Giter))

        if momentum:
            new_change = update + delta_momentum * change
            beta -= new_change
            change = new_change
        else:
            beta -= update

    return beta, cost

#with mini batches
def Adagrad_SGD(X, y, theta, n = 100, n_epochs=50, batch_size=5, eta=0.01, ridge = False, lmbda = 0.01, momentum = False,  delta_momentum=0.3,change = 0.0, Giter=0.0, delta=1e-8):
    M = batch_size
    m = int(n / M)
    cost = []

    for epoch in range(n_epochs):
        for i in range(m):
            random_index = M * np.random.randint(m)
            xi = X[random_index:random_index + M]
            yi = y[random_index:random_index + M]

            if ridge == False:
                gradients = (2.0 / M) * xi.T @ (xi @ theta - yi)
                cost.append(CostOLS(theta))
            else:
                gradients = (2.0 / M) * xi.T @ (xi @ theta - yi) + 2 * lmbda * theta
                cost.append(CostRidge(theta, lmbda))

            Giter += gradients**2
            update = eta * gradients / (delta + np.sqrt(Giter))

            if momentum:
                new_change = update + delta_momentum * change
                theta -= new_change
                change = new_change
            else:
                theta -= update

    return theta, cost


#Taken from week 39 - RMSprop for adaptive learning rate with Stochastic Gradient Descent
#RMSprop_SGD with mini batches and momentum
def RMSprop_SGD(X, y, theta, n = 100,  n_epochs=100, batch_size=5, eta=0.01, ridge = False,lmbda=0.01, rho=0.99, delta=1e-8, Giter=0.0):
    M = batch_size
    m = int(n / M)
    cost = []

    for epoch in range(n_epochs):
        for i in range(m):
            random_index = M * np.random.randint(m)
            xi = X[random_index:random_index + M]
            yi = y[random_index:random_index + M]

            if ridge == False:
                gradients = (2.0 / M) * xi.T @ (xi @ theta - yi)
                cost.append(CostOLS(theta))
            else:
                gradients = (2.0 / M) * xi.T @ (xi @ theta - yi) + 2 * lmbda * theta
                cost.append(CostRidge(theta, lmbda))

            Giter = rho * Giter + (1 - rho) * gradients**2
            update = eta * gradients / (delta + np.sqrt(Giter))
            theta -= update

    return theta, cost

#Taken from week 39 -> And finally ADAM
def Adam_SGD(X, y, theta, n= 100,  n_epochs=100, batch_size=5, eta=0.01, ridge = False, lmbda = 0.01,  beta1=0.9, beta2=0.999, delta=1e-8,iter=0):
    M = batch_size
    m = int(n / M)
    cost = []
    first_moment = 0.0
    second_moment = 0.0

    for epoch in range(n_epochs):
        iter += 1
        for i in range(m):
            random_index = M * np.random.randint(m)
            xi = X[random_index:random_index + M]
            yi = y[random_index:random_index + M]


            if ridge == False:
                gradients = (2.0 / M) * xi.T @ (xi @ theta - yi)
                cost.append(CostOLS(theta))
            else:
                gradients = (2.0 / M) * xi.T @ (xi @ theta - yi) + 2 * lmbda * theta
                cost.append(CostRidge(theta, lmbda))


            first_moment = beta1 * first_moment + (1 - beta1) * gradients
            second_moment = beta2 * second_moment + (1 - beta2) * gradients**2
            first_unbias = first_moment / (1 - beta1**iter)
            second_unbias = second_moment / (1 - beta2**iter)
            update = eta * first_unbias / (np.sqrt(second_unbias) + delta)
            theta -= update

    return theta, cost

""" 
Things to look at:
* For Ridge regression you need now to study the results as functions of the hyper-parameter lambda and the learning rate 
Discuss your results. Recommend seaborn to look at learning rate and lambda.

* A plain gradient descent with a fixed learning rate (you will need to tune it) using the analytical expression for the gradient.
* Add momentum to the plain GD code and compare convergence with a fixed learning rate (you may need to tune the learning rate).
 
* Repeat these steps for stochastic gradient descent with mini batches and a given number of epochs. Use a tunable learning rate as discussed in the lectures from weeks 39 and 40. Discuss the results as functions of the various parameters (size of batches, number of epochs etc).* 

* Effect of adagrad (Trond
#Tror ikke vi trenger å se på GD for disse. Kanskje kun SGD med mini-batches.
* Effect of RMSprop and Adam (trond

* Replace thereafter your analytical gradient with either Autograd or JAX -> compare results 



"""

np.random.seed(2014)
n = 100 #number of data points
x = np.random.rand(n,1) #input
y = 2 + 3*x + 4*x**2 + 0.1*np.random.randn(n,1) #4x^2 + 3x + 2 + noise

p = 2 #polynomial degree

X = np.c_[np.ones((n,1)), x]#design matrix
beta = np.random.randn(2,1) #initialize parameters

#same as project 1
learning_rates = np.logspace(-4, -1, 4)#learning rates
lambdas = np.logspace(-4, 0, 5)# lambdas

#Similar to the plot from week 39: Code with a Number of Minibatches which varies, analytical gradient
# Plotting cost function with different learning rates
def plot_GD (X, y, beta, eta, lmbda =None):
    beta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y
    xnew = np.array([[0],[2]])
    xbnew = np.c_[np.ones((2,1)), xnew]
    ypredict = xbnew.dot(beta)
    ypredict2 = xbnew.dot(beta_linreg)
    plt.plot(xnew, ypredict, "r-")
    plt.plot(xnew, ypredict2, "b-") #blue is the reference -> need to adjust
    plt.plot(x, y ,'ro')
    plt.axis([0,2.0,0, 15.0])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plot_title = (r'Gradient descent eta={}'.format(eta))
    if lmbda is not None:
        plot_title += (r' with Ridge λ={}'.format(lmbda))
    plt.title(plot_title)
    plt.show()


print("Plain Gradient Descent: OLS")
for eta in learning_rates:
    beta_GD, _ = GD(X, y, beta, eta=eta)
    plot_GD(X, y, beta_GD, eta)
"""
print("Gradient Descent with Ridge: Testing Different Lambda Values")
for lmbda in lambdas:
    for eta in learning_rates:
        beta_GD, _ = GD(X, y, beta, eta=eta, ridge=True, lmbda=lmbda)
        plot_GD(X, y, beta_GD, eta, lmbda=lmbda)
"""