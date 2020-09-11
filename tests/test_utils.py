import pytest
import numpy as np
from conflict_model import utils



def test_create_artificial_Y():

    Y = [1, 0, 0, 0, 0, 1]
    
    Y_r = utils.create_artificial_Y(Y)

    assert len(np.where(Y_r != 0)[0]) == len(np.where(Y != 0)[0])