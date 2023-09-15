import pytest
import configparser
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn import preprocessing, model_selection
from copro import conflict, machine_learning

def create_fake_config():

    config = configparser.ConfigParser()

    config.add_section('general')
    config.set('general', 'verbose', str(False))
    config.add_section('machine_learning')
    config.set('machine_learning', 'train_fraction', str(0.7))

    return config

def test_split_scale_train_test_split():

    X1 = [1, 2, 3, 4]
    X2 = [1, 2, 3, 4]
    X3 = [[1, 2], [3, 4], [1, 2], [5, 6]]

    X = np.column_stack((X1, X2, X3))
    Y = [1, 0, 0, 1]
    config = create_fake_config()
    scaler = preprocessing.QuantileTransformer()

    X_train, X_test, y_train, y_test, X_train_ID, X_test_ID = machine_learning.split_scale_train_test_split(X, Y, config, scaler) # delete X_train_geom, X_test_geom

    assert (len(X_train) + len(X_test)) == len(X)