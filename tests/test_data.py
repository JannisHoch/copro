import pytest
import configparser
import numpy as np
import pandas as pd
from conflict_model import data

def create_fake_config():

    config = configparser.ConfigParser()

    config.add_section('general')
    config.set('general', 'verbose', str(False))

    return config

def test_split_XY_data():

    config = create_fake_config()

    X_arr = [[1, 2], [3, 4], [1, 2], [5, 6]]
    y_arr = [1, 0, 0, 1]

    XY_in = np.column_stack((X_arr, y_arr))
    
    X, Y = data.split_XY_data(XY_in, config)

    XY_out = np.column_stack((X, Y))

    XY_false = np.where(np.equal(XY_in, XY_out) == False)[0]

    assert XY_false.size == 0
