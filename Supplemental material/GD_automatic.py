import numpy as np
from functions import *

def GD_auto(X,y,betainit,niterations,eta,lmbda=0,gamma=0):

    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)

    for i in range(niterations):
                    betaold = np.copy(beta)
                    g = gradient(X,y,beta,lmbda)
                    v = gamma*v+eta*g(beta)
                    beta -= v

                    if np.linalg.norm(beta-betaold) < 1e-5:
                            print(f"Stopped after {i+1} iterations")
                            return beta
    

    print(f"Stopped after {niterations} iterations")
    return beta

def GD_Adagrad_auto(X,y,betainit,niterations,eta,delta=1e-7,lmbda=0,gamma=0):
    
    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)
    r = np.zeros(beta.shape)

    for i in range(niterations):
                    betaold = np.copy(beta)
                    g = gradient(X,y,beta,lmbda)
                    
                    r += g(beta)*g(beta)
                    v = gamma*v+(eta/(delta+np.sqrt(r)))*g(beta)
                    beta -= v

                    if np.linalg.norm(beta-betaold) < 1e-5:
                            print(f"Stopped after {i+1} iterations")
                            return beta
    
    print(f"Stopped after {niterations} iterations")
    return beta


def GD_RMSprop_auto(X,y,betainit,niterations,eta,rho=0.9,delta=1e-6,lmbda=0,gamma=0):

    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)
    r = np.zeros(beta.shape)

    for i in range(niterations):
                    betaold = np.copy(beta)
                    g = gradient(X,y,beta,lmbda)
                    
                    r = rho*r+(1-rho)*g(beta)*g(beta)
                    v = gamma*v+(eta/(np.sqrt(delta+r)))*g(beta)
                
                    beta -= v

                    if np.linalg.norm(beta-betaold) < 1e-5:
                            print(f"Stopped after {i+1} iterations")
                            return beta
    
    print(f"Stopped after {niterations} iterations")
    return beta


def GD_ADAM_auto(X,y,betainit,niterations,eta,rho1=0.9,rho2=0.999,delta=1e-8,lmbda=0):

    v = 0
    n = X.shape[0]

    beta = np.copy(betainit)

    r = np.zeros(beta.shape)
    s = np.zeros(beta.shape)


    for i in range(niterations):
                    betaold = np.copy(beta)
                    g = gradient(X,y,beta,lmbda)
                    
                    s = rho1*s+(1-rho1)*g(beta)
                    r = rho2*r+(1-rho2)*g(beta)*g(beta)

                    s_hat = s/(1-rho1**(i+1))
                    r_hat = r/(1-rho2**(i+1))

                    v = eta*(s_hat/(delta+np.sqrt(r_hat)))
                    
                    beta -= v

                    if np.linalg.norm(beta-betaold) < 1e-5:
                            print(f"Stopped after {i+1} iterations")
                            return beta
    

    print(f"Stopped after {niterations} iterations")
    return beta