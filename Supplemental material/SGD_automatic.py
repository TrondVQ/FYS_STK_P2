import numpy as np
from functions import *

def SGD_auto(X,y,betainit,nepochs,nbatches,eta,lmbda=0,gamma=0):

    np.random.seed(2014)

    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)
    betaold = np.zeros((np.shape(beta)))

    ind = np.arange(n)
    np.random.shuffle(ind)

    batch = np.array_split(ind,nbatches)

    for epoch in range(nepochs):
            if np.linalg.norm(beta-betaold) < 1e-5:
                    print(f"Stopped after {epoch} epochs")
                    return beta
            
            for k in range(nbatches):
                betaold = np.copy(beta)

                xk = X[batch[k]]
                yk = y[batch[k]]

                M = len(yk)
            
                g = gradient(xk,yk,beta,lmbda)
                v = gamma*v+eta*g(beta)
                beta -= v
                
    print(f"Stopped after {nepochs} epochs")
    return beta


def SGD_Adagrad_auto(X,y,betainit,nepochs,nbatches,eta,delta=1e-7,lmbda=0,gamma=0):

    np.random.seed(2014)

    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)
    betaold = np.zeros((np.shape(beta)))
    r = np.zeros(beta.shape)


    ind = np.arange(n)
    np.random.shuffle(ind)

    batch = np.array_split(ind,nbatches)
    

    for epoch in range(nepochs):
            if np.linalg.norm(beta-betaold) < 1e-5:
                 print(f"Stopped after {epoch} epochs")
                 return beta
     
            for k in range(nbatches):

                betaold = np.copy(beta)
            
                xk = X[batch[k]]
                yk = y[batch[k]]

                M = len(yk)
            
                g = gradient(xk,yk,beta,lmbda)
                r += g(beta)*g(beta)
                v = gamma*v+(eta/(delta+np.sqrt(r)))*g(beta)
                beta -= v
    
    print(f"Stopped after {nepochs} epochs")
    return beta
  

def SGD_RMSprop_auto(X,y,betainit,nepochs,nbatches,eta,rho=0.9,delta=1e-7,lmbda=0,gamma=0):

    np.random.seed(2014)

    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)
    betaold = np.zeros((np.shape(beta)))
    r = np.zeros(beta.shape)


    ind = np.arange(n)
    np.random.shuffle(ind)

    batch = np.array_split(ind,nbatches)
    

    for epoch in range(nepochs):
            if np.linalg.norm(beta-betaold) < 1e-5:
                 print(f"Stopped after {epoch} epochs")
                 return beta
            
            
            for k in range(nbatches):

                betaold = np.copy(beta)
            
                xk = X[batch[k]]
                yk = y[batch[k]]

                M = len(yk)
            
                g = gradient(xk,yk,beta,lmbda)
                r = rho*r+(1-rho)*g(beta)*g(beta)
                v = gamma*v+(eta/(np.sqrt(delta+r)))*g(beta)
                beta -= v
    

    print(f"Stopped after {nepochs} epochs")
    return beta


def SGD_ADAM_auto(X,y,betainit,nepochs,nbatches,eta=0.001,rho1=0.9,rho2=0.999,delta=1e-8,lmbda=0):

    np.random.seed(2014)

    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)
    betaold = np.zeros((np.shape(beta)))

    r = np.zeros(beta.shape)
    s = np.zeros(beta.shape)


    ind = np.arange(n)
    np.random.shuffle(ind)

    batch = np.array_split(ind,nbatches)
    
    t = 0
    for epoch in range(nepochs):
            if np.linalg.norm(beta-betaold) < 1e-5:
                 print(f"Stopped after {epoch} epochs")
                 return beta
            
            
            t+=1
            for k in range(nbatches):

                betaold = np.copy(beta)
            
                xk = X[batch[k]]
                yk = y[batch[k]]

                M = len(yk)
            
                g = gradient(xk,yk,beta,lmbda)

                s = rho1*s+(1-rho1)*g(beta)
                r = rho2*r+(1-rho2)*g(beta)*g(beta)

                s_hat = s/(1-rho1**(t))
                r_hat = r/(1-rho2**(t))
                
                v = eta*(s_hat/(delta+np.sqrt(r_hat)))
                beta -= v
    

    print(f"Stopped after {nepochs} epochs")
    return beta