import pytest
import numpy as np 
import torch
from torch import optim
from torch.nn import functional as F
from numpy.testing import assert_allclose

from liftorch.divergences import divergence

def argmin(X, fn, variable = None):
    """
    Return argmin_{Z} fn(Z, X). Only used for testing
    """
    n_max_steps_without_improvement = 30
    n_no_improved = 0
    if variable is None:
        variable = torch.zeros(X.shape, requires_grad=True)
        torch.nn.init.normal_(variable)
    optimizer = optim.Adam([variable], lr=0.001)
    tol = 10e-9
    loss = np.inf
    while True:
        optimizer.zero_grad()
        div, (x_min, x_max) = fn(variable, X)
        div.backward()
        optimizer.step()
        with torch.no_grad():
            if x_min is not None or x_max is not None:
                variable.clamp_(min=x_min, max=x_max)

        if div.item() > loss - tol:
            if n_no_improved > n_max_steps_without_improvement:
                return variable
            else:
                n_no_improved +=1
        else:
            loss = div.item()
            n_no_improved = 0

def test_id():
    """
    test if id(X) == argmin_{Z} id_div(Z)
    """
    some_tensor = torch.tensor([[-1., 1., -1.], [2., -3., 4.]])
    Z = argmin(some_tensor, lambda x,z: divergence('id', x, z))
    assert_allclose(Z.detach().numpy(), some_tensor.detach().numpy(), atol=10e-4)

def test_relu():
    """
    test if relu(X) == argmin_{Z>0} relu_div(Z)
    """
    some_tensor = torch.tensor([[-1., -1., 1.], [2., -3., 4.]])
    Z_0 = torch.empty(some_tensor.shape, requires_grad=True)
    torch.nn.init.constant_(Z_0, 0.5)
    Z = argmin(some_tensor, lambda x,z: divergence('relu', x, z), Z_0)
    assert_allclose(Z.detach().numpy(), F.relu(some_tensor).detach().numpy(), atol=10e-3)

def test_sigmoid():
    """
    test if sigmoid(X) == argmin_{0<Z<1} sigmoid_div(Z)
    """
    some_tensor = torch.tensor([[-1., -1., 1.], [2., -3., 4.]])
    Z_0 = torch.empty(some_tensor.shape, requires_grad=True)
    torch.nn.init.constant_(Z_0, 0.5)
    Z = argmin(some_tensor, lambda x,z: divergence('sigmoid', x, z), Z_0)
    assert_allclose(Z.detach().numpy(), F.sigmoid(some_tensor).detach().numpy(), atol=10e-4)

def test_tanh():
    """
    test if tanh(X) == argmin_{-1<Z<1} arctahn_div(Z)
    """
    some_tensor = torch.tensor([[-1., -1., 1.], [2., -3., 4.]])
    Z_0 = torch.empty(some_tensor.shape, requires_grad=True)
    torch.nn.init.constant_(Z_0, 0.1)
    Z = argmin(some_tensor, lambda x,z: divergence('tanh', x, z), Z_0)
    assert_allclose(Z.detach().numpy(), F.tanh(some_tensor).detach().numpy(), atol=10e-4)