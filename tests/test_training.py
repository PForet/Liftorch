from liftorch.modules import LiftedModule

import numpy as np 
from sklearn.metrics import accuracy_score
from torch import nn 
from torch import optim
import torch

class _generator:
    def __init__(self, n_dims=2, seed=1):
        self.n_dims = n_dims
        self.rng = np.random.RandomState(seed=seed)
    
    def get_label(self, x):
        raise NotImplementedError("To override in child class")

    def batch(self, size):
        X = self.rng.rand(size, self.n_dims)
        X = 2*X - 1
        y = [self.get_label(x) for x in X]
        X,y = torch.tensor(X).float(), torch.tensor(y).long()
        return X,y

class LinearlySeparable(_generator):
    """
    Generate linearly separable data (unif in the unit cube).
    Used for testing
    """
    def get_label(self, x):
        return int(x[0] > 0)

def test_training1():
    """
    Optimize the whole loss of the network to check that it trains well.
    Seem dependent on the initial weights of the layers.
    """
    torch.manual_seed(2)
    input_dim = 10
    batch_size = 5000

    class model(LiftedModule):
        def __init__(self):
            super(model, self).__init__()
            self.layer1 = nn.Linear(input_dim,6)
            self.layer2 = nn.Linear(6,2)
            self.set_graph({
                'layer1':'relu',
                'layer2':'id'
            })
        def forward(self, inputs):
            inputs = nn.functional.relu(self.layer1(inputs))
            inputs = nn.functional.log_softmax(self.layer2(inputs), dim=1)
            return inputs
    
    mymodel = model()
    dataset = LinearlySeparable(input_dim, seed=1)
    X,y = dataset.batch(batch_size)

    mymodel.set_batch_size(batch_size)
    mymodel.initialize_activations()
    optimizer = optim.Adam(mymodel.all_parameters(), lr=0.05)
    current_loss = np.inf 

    while True:
        optimizer.zero_grad()
        loss, domains = mymodel.get_lifted_loss(X,y,lambd=0.0002)
        loss.backward()
        optimizer.step()

        mymodel.project_activations(domains)
        if loss.item() > current_loss - 10e-6:
            break 
        current_loss = loss.item()
    
    pred = mymodel(X).max(1, keepdim=True)[1]
    acc = accuracy_score(y, pred)
    assert acc > .95


        