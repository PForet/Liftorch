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
        Base module to create lifted neural networks. Behave live a standard torch.nn.module
        object, but allows the computation of the losses for solving the lifted problem.
        Currently: Only feed-forward networks with linear layers are supported.
        """
        super(LiftedModule, self).__init__(**kwargs)
        self._activations_shape = OrderedDict()
        self._is_graph = False
        self.loss_function = nn.functional.cross_entropy
    
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
    
    def set_batch_size(self, batch_size):
        """
        Replace the first dimention of the X_l tensors with the current batch size
        """
        try:
            activation_shapes = self._activations_shape
        except:
            self._compute_activations_shapes()
            activation_shapes = self._activations_shape

        for k in self._activations_shape.keys():
            self._activations_shape[k][0] = batch_size
    
    def initialize_activations(self, distrib=nn.init.normal_):
        """
        Create X_l tensors. Initialize all values (using a normal distribution). 
        Name conventions are as follows:
        Tensor named 'layer_i' will be the output of layer_i, AFTER composition with the 
        activation function.
        Args:
            ditrib: A pytorch in place initializer (such as normal_ or constant_) used to 
            initialize the tensors.
        """
        if any((v[0] == -1 for e,v in self._activations_shape.items())):
            raise ValueError('No batch size has been defined. Call _set_batch_size()')

        self._activations = OrderedDict()
        for name, shape in self._activations_shape.items():
            tens = torch.empty( * shape, requires_grad = True)
            distrib(tens, 0)
            self._activations[name] = tens
    
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
        self._compute_activations_shapes()
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
    
    def get_W_loss(self, layer, inputs=None):
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
        activation_fn = self._fn_graph[layer]
        div, dom = divergence(activation_fn, self._modules[layer](prev_act), nxt_act)

        return div 

    def get_X_loss(self, layer, inputs = None, y = None):
        """
        Compute the loss for the X step, for a given layer. Return also a tuple (xmin, xmax)
        over which X_l can be optimized.
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
            
            label_loss = self.loss_function(act, y)

            return div_layer + label_loss, dom_layer

    def get_lifted_loss(self, inputs, y, lambd):
        """
        Return the total loss of the relaxed problem, for a given penalty parameter. Also
        return a dictionnary containing the domains on which each X_l can be optimized
        """
        loss = 0
        domains = OrderedDict()
        # Add the divergence losses for all layers
        for name in self._layers():
            div, dom = divergence(
                self._fn_graph[name], # Act. function of current layer
                self._modules[name](inputs), # Comp. of last activations by current layer
                self._activations[name] 
            )
            loss += div
            domains[name] = dom
            inputs = self._activations[name]
        # Add the loss related to the labels:
        label_loss = self.loss_function(inputs, y)
        loss = label_loss + lambd * loss
        
        return loss, domains





