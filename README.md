# Liftorch

## About Liftorch:

Liftorch is a Pytorch extension that allows the user to optimize neural networks based on the relaxed formulation proposed by [El Ghaoui et Al](https://arxiv.org/abs/1805.01532). In this framework, we replace the classical optimization problem of training a neural network:

![](https://latex.codecogs.com/svg.latex?\fn_phv&space;\min_{(W_l,b_l)_0^l,&space;(X_l)_1^L}&space;L(Y,&space;X_{L&plus;1})&space;&plus;&space;\sum_{l=0}^L&space;\pi_l(W_l))

![](https://latex.codecogs.com/svg.latex?\fn_phv&space;s.c.)

![](https://latex.codecogs.com/svg.latex?\fn_phv&space;X_{l&plus;1}&space;=&space;\phi_l(W_lX_l&space;&plus;&space;b_l1_m^T)\,&space;l&space;\in&space;[0,...,L])

By its relaxed formulation:

![](https://latex.codecogs.com/svg.latex?\fn_phv&space;\min_{(W_l,b_l)_0^l&space;,&space;(X_l)_1^L}&space;L(Y,&space;X_{L&plus;1})&space;&plus;&space;\sum_{l=0}^L&space;\pi_l(W_l)&space;&plus;&space;\sum_{l=0}^L&space;\lambda_{l&plus;1}&space;D_l(W_lX_l&plus;&space;b_l1_l^T,&space;X_{l&plus;1}))

![](https://latex.codecogs.com/svg.latex?\fn_phv&space;s.c.)

![](https://latex.codecogs.com/svg.latex?\fn_phv&space;X_l&space;\in&space;Dom_l&space;\,&space;l&space;\in&space;[1,&space;...,&space;L-1])

![](https://latex.codecogs.com/svg.latex?\fn_phv&space;X_0&space;=&space;X)

Where ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;L) stands for the loss function, ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;\pi_l) are penalties imposed on the weights, ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;\phi_l) are the activation functions and ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;D_l) their associated divergence on the feasible set ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;Dom_l) . 
The divergences are convex functions such as 

![](https://latex.codecogs.com/svg.latex?\fn_phv&space;\phi_l(X)&space;=&space;arg\min_{Z&space;\in&space;Dom_l}&space;D_l(X,&space;Z))

Thus, when ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;D_l) is minimum, we have ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;X_{l&plus;1}&space;=&space;\phi_l(W_lX_l&space;&plus;&space;b_l1_m^T)) which is the contraint imposed on the classical optimization problem.

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

![](https://latex.codecogs.com/svg.latex?\fn_phv&space;L(Y,&space;X_{L&plus;1})&space;&plus;&space;\sum_{l=0}^L&space;\lambda_{l&plus;1}&space;D_l(W_lX_l&plus;&space;b_l1_l^T,&space;X_{l&plus;1}))

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

As suggested in the original paper, one may wish to update ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;X_l) and ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;W_l) in a block-coordinate fashion, to take advantage of the fact that the relaxed problem is convex and can be parallelised in ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;W_l).  `Liftorch` propose some methods to obtain only the loss related to certain parameters:

### Optimizing on the layer parameters (![](https://latex.codecogs.com/svg.latex?\fn_phv&space;W_l))
`my_model.get_W_loss(layer = 'layer_i', inputs=X)` will return the loss related to the parameters ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;W_i) of the layer named 'layer_i', that is to say, using the previous notations:

![](https://latex.codecogs.com/svg.latex?\fn_phv&space;D_i(W_{i}X_{i}&plus;b_i1^T,&space;X_{i&plus;1}))

Please note that `inputs` is only needed for computing the loss related to the first layer. 
To optimize only on a certain layer, we can pass to our optimizer the following generator: `optimizer = optim.Adam(my_model.W_parameters('layer_i'), lr=0.01)`. To optimize on all the layers, the usual `optimizer = optim.Adam(my_model.parameters(), lr=0.01)` works.

### Optimizing on the activations (![](https://latex.codecogs.com/svg.latex?\fn_phv&space;X_l))

In this case, `my_model.get_X_loss(layer = 'layer_i', inputs=X, y=y)` will return the loss related to the parameters ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;X_i) (which is, in the usual forward pass, the tensor obtained after composition by 'layer_i' and its activation function). In the case of an hidden layer, this loss is:

![](https://latex.codecogs.com/svg.latex?\fn_phv&space;\lambda&space;\times&space;\left(D_{i-1}(X_i,\,&space;W_{i-1}X_{i-1}&space;&plus;&space;b_{i-1}1^T)&space;&plus;&space;D_{i}(W_iX_i&space;&plus;&space;b_i1^T,\,&space;X_{i&plus;1})\right))

For the last layer, this loss becomes:

![](https://latex.codecogs.com/svg.latex?\fn_phv&space;\lambda&space;D_{L-1}(X_L,\,&space;W_{L-1}X_{L-1}&space;&plus;&space;b_{L-1}1^T)&space;&plus;&space;L(W_LX_L&space;&plus;&space;b_L1^T,&space;y))

Please note that for this method, `inputs` is only needed for the first layer, and `y` is only needed for the last.
To optimize only on ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;X_i), we can pass to our optimizer the generator `my_model.X_parameters('layer_i')`. To optimize on all the ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;X_i,&space;i&space;\in&space;[1,...,L]), we can use `my_model.X_parameters()` (same method, without argument).

## Activation functions:

Are currently supported the following activation functions (everything must be understood point-wise):

| ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;\phi_l) | ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;D_l(z,x)) | ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;Dom_l)|
|---|---|---|
| identity | ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;z^2&space;-&space;2xz) | ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;z&space;\in&space;\mathbb{R})|
| `relu` | ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;z^2&space;-&space;2xz) | ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;z&space;\in&space;\mathbb{R}^&plus;)|
| `sigmoid`| ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;z&space;\ln(z)&space;-&space;(1-z)\ln(1-z)&space;-xz)| ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;x&space;\in&space;[0,1]) |
|`tanh`| ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;(1&plus;z)&space;\ln(1&plus;z)&space;&plus;&space;(1-z)\ln(1-z)&space;-2xz) | ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;x&space;\in&space;[-1,1]) |

## Development

This package is in its early development stage. Only feedforward networks with Linear layers are supported. Convolutional layers should come soon. Better algorithms for box-constraint optimization should be implemented, as the projected gradient method seems to have some limits. Heuristics to find ![](https://latex.codecogs.com/svg.latex?\fn_phv&space;\lambda) and methods to make the optimization simpler might come. 

Every contribution is welcomed. For any question or suggestion, don't hesitate to open an issue or email me at Pierre_foret@berkeley.edu 
