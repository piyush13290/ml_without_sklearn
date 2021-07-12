
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

"""
Some notations: 
    d = number of features including W_0
    n = number of observations

"""

def prepare_X(X:np.ndarray, degree:int = 1)-> np.ndarray: 
    '''
    Expands X as per degrees and appends a column of ones in the begining
    Input:
        X: (n*1) Input matrix
        degress: expanding X to number of degrees
    Returns: 
        X_new : (n * d) matrix
    '''
    assert X.ndim == 2

    n = X.shape[0]

    X_new = X.copy()

    if degree>1:
        for d in range(2,degree+1):
            X_new = np.hstack((X_new, X**d))
    
    # append column of ones'
    X_new = np.hstack((np.ones([n,1]), X_new))

    return X_new

def normaliz_data(X):
    '''
    Z- normalized data and array of means and sds to normalize validation data 
    Input: 
        X: (n*d) matrix
    Returns:
        X_norm : n*d , z-normalized data
        mean = np array of means of all columns 
        std = np array of std of all columns 
    '''
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    mean[0] = 0 # the first columns is column of ones. 
    std[0] = 1 # the first columns is column of ones. 

    X_norm = (X - mean)/ std

    return X_norm, mean, std

def w_closedForm(X,y):
    '''
    Finds the optimal w using closed form solution
    
    Input:
        X = 2D array, N*(D+1) data matrix
        y = 1D array, N lenghth vector of y values
    Output:
        w = 1D array, (D+1) lengthe vector 
    '''
    
    w = np.dot(np.linalg.pinv(X),y)
    
    return w

def give_squared_loss_grad(X, y, w, overPred_penalty=1, underPred_penalty=1):
    '''
    Gives squared loss and grandient given X, w and other parameters
    Input:
        X = 2D array; n*d input matrix
        y = 1D array; (n,) output array
        w = 1D array; (), weights array 
        overPred_penalty = [-Inf, Inf] penulty for over prediction 
        underPred_penalty = [-Inf, Inf] penulty for under prediction 
    Returns:
        loss: float = squared loss
        grad: (d*1) array of gradients
    '''
    
    n = X.shape[0]

    # errors 
    e = np.dot(X,w) - y

    # Penalty for Over/ Under Prediction
    penulty_vect = (e>0.).astype(float)
    penulty_vect[penulty_vect==1] = overPred_penalty
    penulty_vect[penulty_vect==0] = underPred_penalty

    # Asymmetric Loss
    asym_e = np.multiply(penulty_vect, e)

    # Normalised Squared Loss
    loss = np.dot(np.transpose(asym_e), asym_e) /(2*n)

    # Gradient 
    grad = (np.dot(X.T, asym_e)) / n

    return loss, grad


def GradDescent_LinReg(X, y, overPred_penalty=1, underPred_penalty=1, lr=0.1 , maxIt = 10000, verbose=False):
    
    '''
    Finds the optimal w using Gradient Descent method
    Input:
        X = 2D array; n*d input matrix
        y = 1D array; (n,) output array
        w = 1D array; (), weights array 
        overPred_penalty = [-Inf, Inf] penulty for over prediction 
        underPred_penalty = [-Inf, Inf] penulty for under prediction 
        lr = learing rate
        maxIt = Maximum Iterations       
    Returns:
        w  = (d*1) array of weights
    '''
    n,d = X.shape

    if verbose:
        itr_data = []

    # initialize W randomly 
    w = np.random.rand(d,1)

    for i in range(maxIt):
        loss, grad = give_squared_loss_grad(X, y, w, overPred_penalty, underPred_penalty)
        
        if verbose:
            itr_data.append(loss[0][0])
        w = w - (lr*grad)
    
    if verbose:
        return itr_data, w

    return w

def find_best_model_plot_results(X_train, y_train, X_val, y_val, 
                            method:str, overPred_penalty=1, underPred_penalty=1, lr=0.1 , maxIt = 10000 ):


    # checking for degrees till 5
    degrees = [i for i in range(1,6)]

    # storing the best model

    min_loss = np.Inf
    best_model = {}
    all_model = {}

    # for plotting 
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(11, 7)
    plt.plot(X_train, y_train, 'k.')

    x_axis_data = np.linspace(min(X_train)-0.1, max(X_train)+-.1, num=100)

    # below loop: for each polynomial degree finds the w*, finds the val loss and plots the fitted curve
    for d in degrees:
        X = prepare_X(X_train, degree=d)
        X_norm, X_mean, X_std = normaliz_data(X)
        if method == 'ClosedForm':
            w = w_closedForm(X_norm, y_train)
        if method == "GradientDescent":
            w = GradDescent_LinReg(X_norm, y_train, overPred_penalty, underPred_penalty, lr, maxIt)
        train_loss = give_squared_loss_grad(X_norm, y_train, w)[0]
        
        # for validation loss
        X_val_prep = prepare_X(X_val, degree=d)
        X_val_norm = (X_val_prep - X_mean) / X_std
        val_loss = give_squared_loss_grad(X_val_norm, y_val, w)[0]
        
        all_model['degree:'+str(d)] = {'train_loss':train_loss, 'val_loss':val_loss, 'w':w}
        
        if val_loss < min_loss:
            min_loss = val_loss
            best_model['degree'] = d
            best_model['w'] = w
            best_model['train_loss'] = train_loss
            best_model['val_loss'] = val_loss
            
            if method == "GradientDescent":
                best_model['OverPred Penalty'] = overPred_penalty
                best_model['UnderPred Penalty'] = underPred_penalty
                best_model['learning_rate'] = lr
                best_model['maxIt'] : maxIt
        
        # to plot the line
        X_graph = ((prepare_X(x_axis_data, degree=d))-X_mean) / X_std
        
        y_axis_data = np.dot(X_graph,w)
        
        color = {1:'b', 2:'g', 3:'r', 4:'y', 5:'c'}[d]
        plt.plot(x_axis_data, y_axis_data, color, label = "deg:"+str(d))
        
    plt.legend()

    return best_model