import torch 
import numpy
from torch import log1p, log

def divergence(fn_name, Z, X):
    """
    Return the divergence between the tensors for a given activation function, and its domain.
    Current problem: Only works pointwise activaton function. For other 
    activation functions, no efficient Pytorch implementation is possible (we would
    need to sum over the data points, and using Trace is out of question).
    We may have to write specialized CUDA kernels for that.
    """
    if fn_name == 'relu':
        loss, domain = relu_div(Z,X)
    elif fn_name == 'id': 
        loss, domain = id_div(Z,X)
    elif fn_name == 'sigmoid':
        loss, domain = sigmoid_div(Z,X)
    elif fn_name == 'tanh':
        loss, domain = tanh_div(Z,X)
    else:
        raise ValueError('Divergence not implemented for {}.'.format(fn_name))
    if numpy.isinf(loss.item()):
        raise ValueError('Obtained inf loss for divergence {} and tensors {} and  {}'.format(fn_name, Z, X))
    return loss, domain

def relu_div(Z,X):
    """
    Divergence for ReLU:
    div = z.T * z - x.T * z 
    The argmin is the same as the squared Frobenius norm, so we use it instead for efficiency
    """
    loss = (Z-X).norm(2).pow(2)
    domain = (0, None)
    return loss, domain

def id_div(Z,X):
    """
    Divergence for identity function. Same as ReLU but not constrained on the positive real line
    """
    loss = (Z-X).norm(2).pow(2)
    domain = (None, None)
    return loss, domain

def sigmoid_div(Z,X):
    """
    Divergence for sigmoid. We clip the domain at (tol, 1-tol) instead of 
    (0,1) to keep in in the log domain
    """
    tol = 10e-6
    loss = (Z * log(Z) + (1-Z) * log1p(-Z) - X * Z).sum()
    domain = (tol, 1-tol)
    return loss, domain

def tanh_div(Z,X):
    """
    Divergence for tanh. We clip the domain at (1+tol, 1-tol) instead of 
    (-1,1) to keep in in the log domain
    """
    tol = 10e-6
    loss = ((1-Z) * log1p(-Z) + (1+Z) * log1p(Z) - 2 * X * Z).sum()
    domain = (-1+tol, 1-tol)
    return loss, domain