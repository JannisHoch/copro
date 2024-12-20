import pytest
import configparser
import numpy as np
import pandas as pd
import geopandas as gpd
from copro import conflict, machine_learning


def create_fake_config():

    config = configparser.ConfigParser()

    config.add_section("general")
    config.set("general", "verbose", str(False))

    return config


def test_split_conflict_geom_data():
    # TODO: would like to do this with actual geometry information, but np.equal() does not like this...

    X1 = [1, 2, 3, 4]
    # X2 = [['POINT(-58.66 -34.58)'], ['POINT(-47.91 -15.78)'], ['POINT(-70.66 -33.45)'], ['POINT(-74.08 4.60)']]
    X2 = [1, 2, 3, 4]
    X3 = [[1, 2], [3, 4], [1, 2], [5, 6]]

    X_in = np.column_stack((X1, X2, X3))

    X_ID, X_geom, X_data = machine_learning._split_conflict_geom_data(X_in)

    X_out = np.column_stack((X_ID, X_geom, X_data))

    X_false = np.where(np.equal(X_in, X_out) == False)[0]

    assert X_false.size == 0
