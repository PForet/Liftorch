import pytest
import torch
import numpy as np
from numpy.testing import assert_allclose
from torch import nn
from torch.nn import functional as F
from liftorch.modules import LiftedModule

def test_graph():
    """
    1 ) Test if the layer graph is right.
    2 ) Test the error when missing a layer
    3 ) Test the error when adding a wrong layer
    """
    class model1(LiftedModule):
        def __init__(self):
            super(model1, self).__init__()
            self.layer1 = nn.Linear(1,1)
            self.layer2 = nn.Linear(1,1)
            self.last_layer = nn.Linear(1,1)
            self.set_graph({
                'layer1':'relu',
                'last_layer':'id',
                'layer2':'tanh'
            })
        def test_me(self):
            assert self._next_layer('layer1') == 'layer2'
            assert self._next_layer('layer2') == 'last_layer'
            assert self._prev_layer('last_layer') == 'layer2'
            with pytest.raises(ValueError):
                self._next_layer('last_layer')
            with pytest.raises(ValueError):
                self._prev_layer('layer1') 
    
    class model2(LiftedModule):
        def __init__(self):
            super(model2, self).__init__()
            self.layer1 = nn.Linear(1,1)
            self.layer2 = nn.Linear(1,1)
            self.set_graph({
                'layer1':'relu'
            })

    class model3(LiftedModule):
        def __init__(self):
            super(model3, self).__init__()
            self.layer1 = nn.Linear(1,1)
            self.set_graph({
                'layer1':'relu',
                'layer2':'relu'
            })

    m = model1()
    m.test_me()
    with pytest.raises(ValueError):
        m = model2()
    with pytest.raises(ValueError):
        m = model3()

def test_handling():
    """
    various error handling
    """
    # Initialization of activations without _set_batch_size should explicitely fail
    class m1(LiftedModule):
        def __init__(self):
            super(m1, self).__init__()
            self.layer1 = nn.Linear(1,1)
            self.set_graph({'layer1':'id'})
    with pytest.raises(ValueError):
        m = m1()
        m._compute_activations_shapes()
        m.initialize_activations()
    m = m1()
    m._compute_activations_shapes()
    m.set_batch_size(10)
    m.initialize_activations()  


def test_X_optim():
    """
    Make sure that minimizing the X loss of a layer gives the same activation as
    a standard feed forward pass.
    NB: There is a known bug where mixing the activation functions leads to a non-convergence on
    some elements, and a good convergence on others. In any case, the convergence
    is very approximative. We might need a better opt algo
    """
    torch.manual_seed(0)

    class model(LiftedModule):
        def __init__(self):
            super(model, self).__init__()

        def test_first_layer(self, inputs):
            """ 
            Check if the activation of the first hidden layer after a Z optim matches 
            the ones obtained with the forward pass.
            """
            self._compute_activations_shapes()
            self.set_batch_size(2)
            self.initialize_activations()

            with torch.no_grad():
                self._activations['layer2'] = torch.tensor(self.forward_activations[1])

            optimizer = torch.optim.Adam(self.X_parameters('layer1'), lr=0.0005)
            curloss, tol = np.inf, 10e-6
            while True:
                optimizer.zero_grad()
                loss, (xmin, xmax) = self.get_X_loss('layer1', inputs = inputs)
                loss.backward()
                optimizer.step()
                if xmin is not None or xmax is not None:
                    with torch.no_grad():
                        self._activations['layer1'].clamp_(min=xmin, max=xmax)
                if loss.item() > curloss - tol:
                    break
                curloss = loss.item()
            assert_allclose(
                self._activations['layer1'].detach(), 
                self.forward_activations[0].detach(), atol=10e-2)

        def test_second_layer(self):
            """ 
            Check if the activation of a hidden (not first) layer after a Z optim matches 
            the ones obtained with the forward pass. We must set the activations of the first and 
            last layer to the forward activation to get the expected result.
            We have to make sure that we optimize only on the second activation.
            """
            self._compute_activations_shapes()
            self.set_batch_size(2)
            self.initialize_activations()
            optimizer = torch.optim.Adam(self.X_parameters('layer2'), lr=0.0005)

            with torch.no_grad():
                self._activations['layer1'] = torch.tensor(self.forward_activations[0])
                self._activations['layer3'] = torch.tensor(self.forward_activations[2])

            curloss, tol = np.inf, 10e-8
            while True:
                optimizer.zero_grad()
                loss, (xmin, xmax) = self.get_X_loss('layer2', inputs = inputs)
                loss.backward()
                optimizer.step()
                if xmin is not None or xmax is not None:
                    with torch.no_grad():
                        self._activations['layer2'].clamp_(min=xmin, max=xmax)
                if loss.item() > curloss - tol:
                    break
                curloss = loss.item()
            assert_allclose(
                self._activations['layer2'].detach(), 
                self.forward_activations[1].detach(), atol=10e-2)
    
    class model1(model):
        def __init__(self):
            super(model1, self).__init__()
            self.layer1 = nn.Linear(3,5)
            self.layer2 = nn.Linear(5,4)
            self.layer3 = nn.Linear(4,3)
            self.set_graph({
                'layer1':'relu',
                'layer2':'relu',
                'layer3':'id'
            })

        def forward(self, x):
            """
            Standard forward pass. Keep in memory the activations obtained
            """
            self.forward_activations = []
            x = F.relu(self.layer1(x))
            self.forward_activations.append(x)
            x = F.relu(self.layer2(x))
            self.forward_activations.append(x)
            x = self.layer3(x)
            self.forward_activations.append(x)
        
    inputs = torch.tensor([[.34, -0.48, 0.98], [-0.32, .74, .23]])

     # Test for a relu-relu network
    m = model1()
    m.forward(inputs)
    m.test_first_layer(inputs)

    m = model1()
    m.forward(inputs)
    m.test_second_layer()

def test_loss():
    """
    Test if all types of loss are accessible, and check the domains. Does not
    check the loss values.
    """
    class model(LiftedModule):
        def __init__(self):
            super(model, self).__init__()
            self.layer1 = nn.Linear(3,10)
            self.layer2 = nn.Linear(10,6)
            self.layer3 = nn.Linear(6,4)
            self.layer4 = nn.Linear(4,2)
            self.set_graph({
                'layer1':'relu',
                'layer2':'tanh',
                'layer3':'sigmoid',
                'layer4':'id'
            })

    m = model()
    inputs = torch.tensor([[.34, -0.48, 0.98], [-0.32, .74, .23]])
    y = torch.tensor([1,0])

    # Should fail if no batch size was provided:
    with pytest.raises(AttributeError):
        m.get_X_loss('layer1', inputs)
    with pytest.raises(AttributeError):
        m.get_W_loss('layer1', inputs)
    with pytest.raises(AttributeError):
        m.get_lifted_loss('layer1', inputs, y)

    # Now should work
    m.set_batch_size(2)
    m.initialize_activations()
    
    assert not np.isinf(m.get_W_loss('layer2').item())
    assert not np.isinf(m.get_W_loss('layer3').item())
    assert not np.isinf(m.get_W_loss('layer4').item())

    # Explicitely ask for inputs if first layer:
    with pytest.raises(ValueError):
        m.get_W_loss('layer1').item()
    assert not np.isinf(m.get_W_loss('layer1', inputs).item())

    assert not np.isinf(m.get_X_loss('layer2')[0].item())
    assert not np.isinf(m.get_X_loss('layer3')[0].item())

    # Explicitely ask for inputs if first layer:
    with pytest.raises(ValueError):
        m.get_X_loss('layer1').item()
    assert not np.isinf(m.get_X_loss('layer1', inputs)[0].item())

    # Explicitely ask for labels if last layer:
    with pytest.raises(ValueError):
        m.get_X_loss('layer4')[0].item()
    assert not np.isinf(m.get_X_loss('layer4', y=y)[0].item())

    # Check domain
    _, dom = m.get_X_loss('layer1', inputs)
    assert dom[0] == 0 and dom[1] is None # For Relu

    loss, domain = m.get_lifted_loss(inputs, y, lambd = 1.)
    assert not np.isinf(loss.item())
    assert domain['layer1'][0] == 0
    assert domain['layer1'][1] is None
    assert domain['layer4'][0] is None

