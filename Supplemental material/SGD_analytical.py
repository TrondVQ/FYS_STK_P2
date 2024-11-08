import numpy as np

def SGD(X,y,betainit,nepochs,nbatches,eta,lmbda=0,gamma=0):
    """
    Stochastic gradient descent with a constant learning rate and optional momentum

    Args:
        X (ndarray): design matrix
        y (ndarray): target values
        betainit (ndarray): inital regression paramteres
        nepochs (int): number of epochs
        nbatches (int): number of mini-batches
        eta (float): learning rate
        lmbda (float): hyperparameter
        gamma (float): momentum(friction) parameter

    Returns:
        beta (ndarray): estimated optimal parameters

    """

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
            
                g = (2.0/M)*xk.T @ (xk @ (beta)-yk)+2*lmbda*beta
                v = gamma*v+eta*g
                beta -= v
                
    print(f"Stopped after {nepochs} epochs")
    return beta


def SGD_Adagrad(X,y,betainit,nepochs,nbatches,eta,delta=1e-7,lmbda=0,gamma=0):
    """
    Stochastic gradient descent with Adagrad and optional momentum

    Args:
        X (ndarray): design matrix
        y (ndarray): target values
        betainit (ndarray): inital regression paramteres
        nepochs (int): number of epochs
        nbatches (int): number of mini-batches
        eta (float): global learning rate
        delta (float): small value for numerical stability
        lmbda (float): hyperparameter
        gamma (float): momentum(friction) parameter

    Returns:
        beta (ndarray): estimated optimal parameters
    """

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
            
                g = (2.0/M)*xk.T @ (xk @ (beta)-yk)+2*lmbda*beta
                r += g*g
                v = gamma*v+(eta/(delta+np.sqrt(r)))*g
                beta -= v
    

    print(f"Stopped after {nepochs} epochs")
    return beta
  

def SGD_RMSprop(X,y,betainit,nepochs,nbatches,eta,rho=0.9,delta=1e-7,lmbda=0,gamma=0):
    """
    Stochastic gradient descent with RMSProp and optional momentum

    Args:
        X (ndarray): design matrix
        y (ndarray): target values
        betainit (ndarray): inital regression paramteres
        nepochs (int): number of epochs
        nbatches (int): number of mini-batches
        eta (float): global learning rate
        rho (float): decay rate
        delta (float): small value for numerical stability
        lmbda (float): hyperparameter
        gamma (float): momentum(friction) parameter

    Returns:
        beta (ndarray): estimated optimal parameters
    """ 

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
            
                g = (2.0/M)*xk.T @ (xk @ (beta)-yk)+2*lmbda*beta
                r = rho*r+(1-rho)*g*g
                v = gamma*v+(eta/(np.sqrt(delta+r)))*g
                beta -= v
    

    print(f"Stopped after {nepochs} epochs")
    return beta


def SGD_ADAM(X,y,betainit,nepochs,nbatches,eta=0.001,rho1=0.9,rho2=0.999,delta=1e-8,lmbda=0):
    """
    Stochastic gradient descent with Adam 

    Args:
        X (ndarray): design matrix
        y (ndarray): target values
        betainit (ndarray): inital regression paramteres
        nepochs (int): number of epochs
        nbatches (int): number of mini-batches
        eta (float): global learning rate
        rho1(float): decay rate for first moment
        rho2(float): decay rate for second moment
        delta (float): small value for numerical stability
        lmbda (float): hyperparameter
        gamma (float): momentum(friction) parameter

    Returns:
        beta (ndarray): estimated optimal parameters
    """ 

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
            
                g = (2.0/M)*xk.T @ (xk @ (beta)-yk)+2*lmbda*beta

                s = rho1*s+(1-rho1)*g
                r = rho2*r+(1-rho2)*g*g

                shat = s/(1-rho1**(t))
                rhat = r/(1-rho2**(t))
                
                v = eta*(shat/(delta+np.sqrt(rhat)))
                beta -= v
    

    print(f"Stopped after {nepochs} epochs")
    return beta