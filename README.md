# Liftorch

## About Liftorch:

Liftorch is a Pytorch extension that allows the user to optimize neural networks based on the relaxed formulation proposed by [El Ghaoui et Al](https://arxiv.org/abs/1805.01532). In this framework, we replace the classical optimization problem of training a neural network:

![](https://latex.codecogs.com/png.latex?$$\min_{(W_l,b_l)_0^l,&space;(X_l)_1^L}&space;L(Y,&space;X_{L&plus;1})&space;&plus;&space;\sum_{l=0}^L&space;\pi_l(W_l)&space;\\&space;s.c.&space;\\&space;X_{l&plus;1}&space;=&space;\phi_l(W_lX_l&space;&plus;&space;b_l1_m^T)\,&space;,&space;\,&space;l&space;\in&space;[0,...,L]&space;\\&space;X_0&space;=&space;X&space;$$)

By its relaxed formulation:

![](https://latex.codecogs.com/png.latex?\fn_phv&space;\begin{array}{c}&space;$$\min_{(W_l,b_l)_0^l&space;,&space;(X_l)_1^L}&space;L(Y,&space;X_{L&plus;1})&space;&plus;&space;\sum_{l=0}^L&space;\pi_l(W_l)&space;&plus;&space;\sum_{l=0}^L&space;\lambda_{l&plus;1}&space;D_l(W_lX_l&plus;&space;b_l1_l^T,&space;X_{l&plus;1})&space;\\&space;s.c.&space;\\&space;X_l&space;\in&space;Dom_l&space;\,&space;,&space;\,&space;l&space;\in&space;[1,&space;...,&space;L-1]&space;\\&space;X_0&space;=&space;X$$&space;\end{array})

Where $L$ stands for the loss function, $\pi_l$ are penalties imposed on the weights, $\phi_l$ are the activation functions and $D_l$ their associated divergence on the feasible set $Dom_l$. 
The divergences are convex functions such as 

$$ \phi_l(X) = arg\min_{Z \in Dom_l} D_l(X, Z) $$

Thus, when $D_l$ is minimum, we have $X_{l+1} = \phi_l(W_lX_l + b_l1_m^T)$ which is the contraint imposed on the classical optimization problem.

This _lifted_ formulation has been shown to provide excellent initial values for the initialization of the layers weights. For this reason, *Liftorch* aims at providing an easy way to solve this optimization problem for feedforward neural networks implemented in *Pytorch*.

## How it works:

*Liftorch* propose an extension of *Pytorch* `torch.nn.Module` class that keeps the same functionnalities while adding methods to solve the relaxed optimization problem. For instance, we consider the case of a 3 layers feed forward classifier with ReLU activation functions and a cross_entropy loss. The *Pytorch* implementation of such a network would be:
```Python
from torch import nn
from torch.nn import functional as F
class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 8)
        self.layer3 = nn.Linear(8, 2)
    def forward(self, inputs):
        inputs = F.relu(self.layer1(inputs))
        inputs = F.relu(self.layer2(inputs))
        return self.layer3(inputs)
```
With *Liftorch*, the implementation is almost the same, at the exception that we have to explicitely declare which activation function we use after each layer:

```Python
from torch import nn
from torch.nn import functional as F
from liftorch.modules import LiftedModule

class classifier(LiftedModule):
    def __init__(self):
        super(classifier, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 8)
        self.layer3 = nn.Linear(8, 2)
        self.set_graph({
            'layer1':'relu',
            'layer2':'relu',
            'layer3':'id',
        })
    def forward(self, inputs):
        inputs = F.relu(self.layer1(inputs))
        inputs = F.relu(self.layer2(inputs))
        return self.layer3(inputs)
```

In *Pytorch*, a basic training would look like:
```Python
# X in a batch of 100 observations of 10 variables each and y are the associated labels

my_model = classifier()

from torch import optim
optimizer = optim.Adam(my_model.parameters(), lr=0.005)
for epoch in range(10):
    optimizer.zero_grad()
    loss = F.cross_entropy(my_model(X), y)
    loss.backward()
    optimizer.step()
````

`Liftorch` allows solving for the relaxed problem, by providing an easy access to the loss

$$ L(Y, X_{L+1}) + \sum_{l=0}^L \lambda_{l+1} D_l(W_lX_l+ b_l1_l^T, X_{l+1})$$
via the `get_lifted_loss(X,y,lambda)` method. This minimization problem can be solve with only a few adjustement to the previous code:

```Python
my_model = classifier()

# Declare the size of the training batch.
my_model.set_batch_size(100)
# Initialize X_l with a gaussian distribution
my_model.initialize_activations(distrib=nn.init.normal_)
# Declare the loss function to uses
my_model.loss_function = F.cross_entropy

# We call all_parameters to optimize over W_l and X_l
optimizer = optim.Adam(my_model.all_parameters(), lr=0.005)

for epoch in range(10):
    optimizer.zero_grad()
    loss, domain = my_model.get_lifted_loss(X, y, lambd=0.1)
    loss.backward()
    optimizer.step()
    # We project the X_l tensors on their domains (Dom_l)
    my_model.project_activations(domain)
```

Once this problem solved, we can return to the code above to fine-tune the optimization, leveraging the weights we obtained by solving the relaxed problem as a very good initialization. We can also use our model as it is to make predictions, using the usual *Pytorch* syntax.

## Other optimization methods:

As suggested in the original paper, one may wish to update $X_l$ and $W_l$ in a block-coordinate fashion, to take advantage of the fact that the relaxed problem is convex and can be parallelised in $W_l$.  `Liftorch` propose some methods to obtain only the loss related to certain parameters:

### Optimizing on the layer parameters ($W_l$)
`my_model.get_W_loss(layer = 'layer_i', inputs=X)` will return the loss related to the parameters $W_i$ of the layer named 'layer_i', that is to say, using the previous notations:

$$  D_i(W_{i}X_{i}+b_i1^T, X_{i+1}) $$

Please note that `inputs` is only needed for computing the loss related to the first layer. 
To optimize only on a certain layer, we can pass to our optimizer the following generator: `optimizer = optim.Adam(my_model.W_parameters('layer_i'), lr=0.01)`. To optimize on all the layers, the usual `optimizer = optim.Adam(my_model.parameters(), lr=0.01)` works.

### Optimizing on the activations ($X_l$)

In this case, `my_model.get_X_loss(layer = 'layer_i', inputs=X, y=y)` will return the loss related to the parameters $X_i$ (which is, in the usual forward pass, the tensor obtained after composition by 'layer_i' and its activation function). In the case of an hidden layer, this loss is:

$$
\lambda  \times \left(D_{i-1}(X_i,\, W_{i-1}X_{i-1} + b_{i-1}1^T) + D_{i}(W_iX_i + b_i1^T,\, X_{i+1})\right)
$$
For the last layer, this loss becomes:

$$
\lambda D_{L-1}(X_L,\, W_{L-1}X_{L-1} + b_{L-1}1^T) + L(W_LX_L + b_L1^T, y)
$$

Please note that for this method, `inputs` is only needed for the first layer, and `y` is only needed for the last.
To optimize only on $X_i$, we can pass to our optimizer the generator `my_model.X_parameters('layer_i')`. To optimize on all the $X_i, i \in [0,..L]$, we can use `my_model.X_parameters()` (same method, without argument).

## Activation functions:

Are currently supported the following activation functions (everything must be understood point-wise):

| $\phi_l$ | $D_l(z,x)$ | $Dom_l$ |
|---|---|---|
| identity | $z^2 - 2xz$ | $z \in \mathbb{R}$|
| `relu` | $z^2 - 2xz$ | $z \in \mathbb{R}^+$|
| `sigmoid`| $z \ln(z) - (1-z)\ln(1-z) -xz$| $x \in [0,1]$ |
|`tanh`| $(1+z) \ln(1+z) + (1-z)\ln(1-z) -2xz$ | $x \in [-1,1]$ |

## Development

This package is in its early development stage. Only feedforward networks with Linear layers are supported. Convolutional layers should come soon. Better algorithms for box-constraint optimization should be implemented, as the projected gradient method seems to have some limits. Heuristics to find $\lambda$ and methods to make the optimization simpler might come. 

Every contribution is welcomed. For any question or suggestion, don't hesitate to open an issue or email me at Pierre_foret@berkeley.edu 
