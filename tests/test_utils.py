import pytest
from conflict_model import utils

Y = [1, 0, 0, 0, 0, 1]

def create_artificial_Y(Y):
    
    Y_r = utils.create_artificial_Y(Y)

    assert len(np.where(Y_r != 0)[0]) == len(np.where(Y != 0)[0])