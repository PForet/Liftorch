import pytest
from torch import nn
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