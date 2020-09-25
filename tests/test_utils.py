import pytest
import numpy as np
import pandas as pd
from conflict_model import utils

def test_create_artificial_Y():

    Y = [1, 0, 0, 0, 0, 1]
    
    Y_r = utils.create_artificial_Y(Y)

    assert len(np.where(Y_r != 0)[0]) == len(np.where(Y != 0)[0])

def test_get_conflict_datapoints_only():

    X_arr = [[1, 2], [3, 4], [1, 2], [5, 6]]
    y_arr = [1, 0, 0, 1]

    X_in = pd.DataFrame(data=X_arr, columns=['var1', 'var2'])
    y_in = pd.DataFrame(data=y_arr, columns=['y_test'])

    X_out, y_out = utils.get_conflict_datapoints_only(X_in, y_in)

    test_arr = np.where(y_out.y_test.values == 0)[0]

    assert test_arr.size == 0