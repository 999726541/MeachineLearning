""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """
    # TODO: Finish this function
    n,m = data.shape
    # total number point is n, feature is m
    data = np.c_[data, [1.0] * n]
    result = []
    for i in range(n):
        p=(np.dot(data[i],weights))[0] # np.c_[data[i],1]==> 1*(M+1); weights==>(M+1)*1

        result.append(sigmoid(p))
    y = np.array([result]).T
    return y


def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    n = len(y)
    predict = np.array((y >= 0.5)).astype(np.int)
    ce = float(np.dot(targets.T,np.log(y))+np.dot((np.array([[1]*n])-targets.T),np.log((np.array([[1]*n]).T-y))))
    ce = float(-ce/n)
    frac_correct = 1.00-float(np.sum(abs(predict-targets)))/float(len(predict))
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """

    y = logistic_predict(weights, data)

    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
    else:
        n, m = data.shape
        data = np.c_[data, [1.0] * n]
        f = 0.0
        zi = []
        for i in range(n):
            z = np.dot(data[i], weights)
            f = (np.log(np.exp(-z)) + 1) + f + (1-targets[i]) * z
            zi.append(float(np.exp(-z) / (1.0 + np.exp(-z))))
        zi = np.array([zi])
        df = []
        for k in range(m+1):
            # Adding regularization into df to prevent over fitting
            ddf = (np.dot(1-targets.T, data.T[k]) - np.dot(np.array([data.T[k],]) , zi.T))
            df.append(ddf[0][0])

        df = np.array([df]).T
    return f, df, y

def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """

    # TODO: Finish this function
    n,m = data.shape
    data = np.c_[data,[1.0]*n]
    f = 0.0
    zi = []
    decay = float(hyperparameters['weight_decay'])
    for i in range(n):
        z = np.dot(data[i], weights)
        f = (np.log(np.exp(-z))+1) + f + (1-targets[i])*z
        zi.append(np.exp(-z) / (1.0 + np.exp(-z)))
    df = []
    for k in range(m+1):
        # Adding regularization into df to prevent over fitting
        df.append((np.dot(1-targets.T,data.T[k])-np.dot(data.T[k],np.array([zi]).T)
                  + decay*weights[k])[0][0])
    df = np.array([df]).T
    
    return f, df

