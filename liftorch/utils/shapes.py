from torch.nn import Module, Linear, Conv3d

def _get_linear_shape(module):
    """
    Return the shape of the output of a linear module. First dimension (batch size) is -1
    """
    return [-1, module.out_features]

def get_output_shape(module):
    """
    Return the shape of the output of a module (a Pytorch layer). Currently only deals with linear modules
    More to add after
    """
    if not isinstance(module, Module):
        raise ValueError("Can't get output shape of object of type {}.".format(type(module)))
    else:
        if isinstance(module, Linear):
            return _get_linear_shape(module)
        elif False: # Add more conditions when more layers type will be supported
            pass 
        else:
            raise ValueError("{} layers are not supported yet.".format(module.__class__.__name__))
