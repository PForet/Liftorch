from collections import OrderedDict

import torch
from torch.nn import Module
from torch import optim
from torch import nn
import numpy as np 
from ..utils.shapes import get_output_shape
from ..divergences import divergence

class LiftedModule(Module):
    def __init__(self, **kwargs):
        """
        Module to create lifted neural networks. Only add functionnalities to classical Pytorch module.
        Anything that works with a Pytorch 'Module' should works the same way with a lifted Module.
        Currently: Only feed-forward networks with linear layers are supported.
        """
        super(LiftedModule, self).__init__(**kwargs)
        self._activations_shape = OrderedDict()
        self._is_graph = False
        self._is_shapes = False
        self.lambd = 10 # To change later, find a way to calibrate it
        self.lifted_loss = nn.functional.cross_entropy # Same
    
    def _compute_activations_shapes(self):
        """
        Fill the _activations_shape dictionnary with the shapes of the X_l tensors. 
        """
        for name, module in self._modules.items():
            self._activations_shape[name] = get_output_shape(module)
        self._is_shapes = True
    
    def _layers(self):
        """
        Return a list containing the names of the different layers
        """
        return list(self._modules.keys())

    def X_parameters(self, layer=None):
        """
        Return an iterator that yields the activation tensors (X_l). If 'layer' is given,
        return only the activation after this layer.
        """
        if layer is None:
            for name, value in self._activations.items():
                yield value
        else:
            yield self._activations[layer]
    
    def _set_batch_size(self, batch_size):
        """
        Replace the first dimention of the X_l tensors with the current batch size
        """
        for k in self._activations_shape.keys():
            self._activations_shape[k][0] = batch_size
    
    def _initialize_activations(self):
        """
        Create X_l tensors. Initialize all values to zero. Name conventions are as follows:
        tensor named 'layer_i' will be the output of layer_i, AFTER composition with the 
        activation function.
        """
        if any((v[0] == -1 for e,v in self._activations_shape.items())):
            raise ValueError('No batch size has been defined. Call _set_batch_size()')

        self._activations = OrderedDict()
        for name, shape in self._activations_shape.items():
            tens = torch.empty( * shape, requires_grad = True)
            nn.init.normal_(tens, 0)
            self._activations[name] = tens
        self._make_default_optimizers() # Create optimizers for the activations

    def _make_default_optimizers(self):
        """
        Create some defaults optimizers for the two optimization steps (X and W). Both can be
        replaced using 'set_X_optimizer' and 'set_W_optimizer'
        """
        self.set_X_optimizer(optim.Adam, {'lr':0.001})
        self.set_W_optimizer(optim.Adam, {'lr':0.001})
    
    def set_graph(self, graph):
        """
        Allows to declare which activation function is used after each layer. For instance, if our neural network
        has two layers named 'layer1' and 'last_layer', and we apply ReLU after the first layer and 'id' 
        after the second, we should have: graph = {'layer1':'relu', 'last_layer':'id'}.
        For classification tasks, the last layer should have the 'id' (identity) transformation, as 
        a log softmax is applied after.
        Args:
            graph: A dictionnary with layer names as key and activation functions as values
        """
        acceptable_functions = ['relu', 'id', 'sigmoid', 'tanh'] # More to add
        for k,v in graph.items():
            if k not in self._modules.keys():
                raise ValueError('No module named {}.'.format(k))
            if v not in acceptable_functions:
                raise ValueError('{} activation function not supported yet.'.format(v))
        for layer_name in self._modules.keys():
            if layer_name not in graph.keys():
                raise ValueError("No activation function set for layer {}.".format(layer_name))
        self._fn_graph = graph
        self._make_adj_graph()
        self._is_graph = True
    
    def _make_adj_graph(self):
        """
        Compute a dictionnary containing, for each layer, the name of the previous and next layers.
        Useful because we often want these values and need them in O(1)
        """
        self._adj_graph = dict()
        previous_layer = None
        for name, nxt_name in zip(self._layers()[:-1], self._layers()[1:]):
            self._adj_graph[name] = {
                'previous':previous_layer,
                'next':nxt_name
            }
            previous_layer = name
        self._adj_graph[self._layers()[-1]] = {
            'previous':previous_layer,
            'next':None
        }

    def _is_first_layer(self, name):
        """
        Return True if 'name' is the name of the first layer of the network
        """
        return name == self._layers()[0] 
    
    def _is_last_layer(self, name):
        """
        Return True if 'name' is the name of the last layer of the network
        """
        return name == self._layers()[-1] 
    
    def _next_layer(self, name):
        """
        Return the name of the next layer in the graph. Raise error if 'name' refers
        to the last layer.
        """
        nxt = self._adj_graph[name]['next']
        if nxt is not None:
            return nxt 
        raise ValueError("No layer after {}".format(name))
    
    def _prev_layer(self, name):
        """
        Return the name of the previous layer in the graph. Raise error if 'name' refers
        to the first layer.
        """
        prv = self._adj_graph[name]['previous']
        if prv is not None:
            return prv 
        raise ValueError("No layer before {}".format(name))

    def set_X_optimizer(self, optimizer, args):
        """
        Replace the default optimizer for the X step (optimization over the activations)
        """
        self._X_opt = optimizer(self.X_parameters(), **args)
    
    def set_W_optimizer(self, optimizer, args):
        """
        Replace the defautl optimizer for the W step (optimization over the weights)
        """
        self._W_opt = optimizer(self.parameters(), **args)
    
    def _get_W_loss(self, layer, inputs=None):
        """
        Compute the loss for the W step. Don't return the domain of the divergence, as it 
        is used only for optimizing the activations, and not the weights.
        Args:
            layer: the name of the layer for which we want to compute the divergence
            inputs: Inputs tensor (the features). Only needed for the first layer
        """
        if self._is_first_layer(layer):
            if inputs is None:
                raise ValueError('inputs should be provided to compute the loss of the first layer')
            else:
                prev_act = inputs
        else:
            prev_act = self._activations[self._prev_layer(layer)]

        nxt_act = self._activations[layer]
        activation_fn = self._fn_graph[name]
        div, dom = divergence(activation_fn, self._modules[layer](prev_act), nxt_act)

        return div 

    def _get_X_loss(self, layer, inputs = None, y = None):
        """
        Compute the loss for the X step. Return a dict of domain on which each X_l can be optimized.
        To do: Maybe split this into subfunctions, it becomes tedious to read
        Problem: How to deal with the conflicts between divergence domain ? I suppose we
        only use the domaine of the divergence 'l-1' (see eq 13 of El Ghaoui paper).
        Args:
            layer: the name of the layer before the activations for which we want to compute the divergence
            inputs: Inputs tensor (the features). Only needed for the first layer
            y: True output tensors (labels). Only needed for the last layer
        """
        
        if not self._is_last_layer(layer): # Last layer is dealt with separatedly as we must add the label loss

            if self._is_first_layer(layer):
                if inputs is not None:
                    prev_act = inputs
                else:
                    raise ValueError('inputs should be provided to compute the loss of the first layer')
            else:
                prev_act = self._activations[self._prev_layer(layer)]
            
            act = self._activations[layer]
            next_act = self._activations[self._next_layer(layer)]

            activation_fn_layer = self._fn_graph[layer]
            activation_fn_next_layer = self._fn_graph[self._next_layer(layer)]

            div_layer, dom_layer = divergence(
                activation_fn_layer, 
                act,
                self._modules[layer](prev_act))

            div_next_layer, dom_next_layer = divergence(
                activation_fn_next_layer,
                self._modules[self._next_layer(layer)](act),
                next_act)

            return div_layer + div_next_layer, dom_layer

        else: # For the last layer:
            if y is None:
                raise ValueError('labels should be provided to compute the loss of the last layer')
            prev_act = self._activations[self._prev_layer(layer)]
            act = self._activations[layer]
            activation_fn_layer = self._fn_graph[layer]

            div_layer, dom_layer = divergence(
                activation_fn_layer, 
                act,
                self._modules[layer](prev_act))
            
            label_loss = self.lifted_loss(act, y)

            return div_layer + label_loss, dom_layer

    def _X_projection(self, domains):
        """
        Project each X tensor on its domain. Doesn't keep track of gradients when doing so.
        Args:
            domains: A dictionnary containing the name and the domain of each activation tensor
        """
        with torch.no_grad():
            for name, (x_inf, x_sup) in domains.items():
                if x_inf is not None and x_sup is not None:
                    self._activations[name].clamp_(min=x_inf, max=x_sup)

    def _W_partial_step(self, inputs):
        """
        Perform an optimization step on the weights, using gradient descent.
        Return the value of the loss.
        Args:
            inputs: Inputs tensor (the features)
        """
        self._W_opt.zero_grad() # Clear gradient from previous steps
        W_loss = self._get_W_loss(inputs)
        W_loss.backward()
        self._W_opt.step()
        return W_loss.item()

    def _X_partial_step(self, inputs, y):
        """
        Perform an optimization step over the activations (X_l in the paper). After each gradient step,
        project the activations on their domains.
        Return the value of the loss.
        Args:
            inputs: Inputs tensor (the features)
            y: True output tensors (labels)
        """
        self._X_opt.zero_grad() # Clear gradient from previous steps
        X_loss, domains = self._get_X_loss(inputs, y)
        X_loss.backward()
        self._X_opt.step()
        self._X_projection(domains)
        return X_loss.item()

    def _W_step(self, inputs, tol=10e-5, max_iter=500):
        """
        Repeat the optimization step over W until the loss doesn't decrease anymore.
        Args:
            inputs: Inputs tensor (the features)
            tol: Minimum required increase at each step
            max_iter: Maximum number of optimization steps allowed
        """
        current_loss = np.inf 
        for _ in range(max_iter):
            new_loss = self._W_partial_step(inputs)
            if new_loss > current_loss - tol:
                break 
            current_loss = new_loss

    def _X_step(self, inputs, y, tol=10e-5, max_iter=500):
        """
        Repeat the optimization step over W until the loss doesn't decrease anymore.
        Args:
            inputs: Inputs tensor (the features)
            y: True output tensors (labels)
            tol: Minimum required increase at each step
            max_iter: Maximum number of optimization steps allowed
        """
        current_loss = np.inf 
        for _ in range(max_iter):
            new_loss = self._X_partial_step(inputs, y)
            if new_loss > current_loss - tol:
                break 
            current_loss = new_loss

    def lifted_training(self, X, y, n_epoch):
        """
        Perform a training step using the lifted formulation.
        Args:
            X: The input tensor
            y: The labels
            n_epoch: Number of epoch (block coordinate steps) to perform
        """
        if not self._is_shapes:
            self._compute_activations_shapes()
        if not self._is_graph:
            raise ValueError("No function graph has been declared using 'set_graph'")
        self._set_batch_size(X.shape[0])
        self._initialize_activations()
        for _ in range(n_epoch):
            self._X_step(X, y)
            self._W_step(X)





