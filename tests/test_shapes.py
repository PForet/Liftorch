import pytest 
from torch.nn import Linear, Conv3d
from liftorch.utils.shapes import get_output_shape

def test():
    """
    Pytest
    """
    not_a_layer = "a string"
    with pytest.raises(ValueError):
        get_output_shape(not_a_layer)

    convlayer = Conv3d(3,16,4)
    with pytest.raises(ValueError):
        get_output_shape(convlayer)

    linear10 = Linear(3,10)
    assert get_output_shape(linear10) == [-1,10]