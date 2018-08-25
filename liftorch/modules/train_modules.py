from collections import OrderedDict
from ..modules import LiftedModule
from ..utils import as_funct
import numpy as np
import torch
from torch import optim


class TLModule(LiftedModule):
    def __init__(self, **kwargs):
        """
        A module that implement functions to allow lifted training. Doesn't rely on the 
        convex block-coordinate descent. Instead, optimize the loss of the relaxed
        problem on X and W together, by gradient descent.
        NB: In dev, currently not tested.
        """
        super(TLModule, self).__init__(**kwargs)
        self._X_opt_caract = None
        self._opt_caract = None
        self._X_opt = None
        self._opt = None
    
    def set_X_opt(self, opt, **kwargs):
        """
        Set the optimizer used for minimizing the loss related to the activations
        """
        self._X_opt_caract = (opt, kwargs)
    
    def set_opt(self, opt, **kwargs):
        """
        Set the optimizer used for minimizing the lifted loss
        """
        self._opt_caract = (opt, kwargs)
    
    def train(self, X, y, lambd, epoch, warmstart=True):
        """
        Perform 'epoch' optimization steps on the lifted loss. W_l and X_l are minimized 
        jointly. 
        Args:
            X: inputs tensor
            y: labels 
            lambd: The penalty parameter.
            epoch: Number of optimization steps to perform.
            warmstart: If True, initialize X_l using a forward pass. Else, perform a first 
            minimization of X_l.
        """
        if self._X_opt_caract is None and not warmstart:
            raise ValueError("""No optimizer has been set for the X step. 
                Set one using 'set_X_opt'or use warmstart instead.""")
        if self._opt_caract is None:
            raise ValueError("No optimizer has been set. Use 'set_opt' to set one.")
        if warmstart:
            self._warmstart(X)
        else:
            self._coldstart(X,y,lambd)

        opt, args = self._opt_caract
        self._opt = opt(self.all_parameters(), **args)

        for _ in range(epoch):
            self._train_step(X,y,lambd)
        
        self._opt = None # Cleaning up

    def _train_step(self, X, y, lambd):
        """
        Perform a unique training step. Suppose the activations are already set
        """
        self._opt.zero_grad()
        loss, dom = self.get_lifted_loss(X, y, lambd)
        loss.backward()
        self._opt.step()
        self.project_activations(dom)

    def _warmstart(self, X):
        """
        Initialize the activations using a forward pass (so that their match the activation
        of a standard neural network)
        """
        self._activations = OrderedDict()

        for name in self._layers():
            activationFunctionName = self._fn_graph[name]
            layer = self._modules[name]
            with torch.no_grad():
                X = as_funct(activationFunctionName)(layer(X))
            X.requires_grad = True # Should check if this is the canonical way to do it
            self._activations[name] = X

    def _coldstart(self, X, y, lambd):
        """
        Randomly initialize the activation functions, then optimize them until convergence.
        """
        self.set_batch_size(X.shape[0])
        self.initialize_activations()

        opt, args = self._X_opt_caract
        self._X_opt = opt(self.X_parameters(), **args)
        
        currentLoss, tol  = np.inf, 10e-8
        while True:
            self._X_opt.zero_grad()
            loss, dom = self.get_lifted_loss(X, y, lambd)
            loss.backward()
            self._X_opt.step()
            self.project_activations(dom)
            if loss.item() > currentLoss - tol:
                break 
            currentLoss = loss.item()
        
        self._X_opt = None # Cleaning up 


