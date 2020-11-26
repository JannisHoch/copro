import pytest
import configparser
import numpy as np
import pandas as pd
import geopandas as gpd
from copro import conflict

def create_fake_config():

    config = configparser.ConfigParser()

    config.add_section('general')
    config.set('general', 'verbose', str(False))

    return config

def test_split_conflict_geom_data():
    #TODO: would like to do this with actual geometry information, but np.equal() does not like this...

    X1 = [1, 2, 3, 4]
    # X2 = [['POINT(-58.66 -34.58)'], ['POINT(-47.91 -15.78)'], ['POINT(-70.66 -33.45)'], ['POINT(-74.08 4.60)']]
    X2 = [1, 2, 3, 4]
    X3 = [[1, 2], [3, 4], [1, 2], [5, 6]]

    X_in = np.column_stack((X1, X2, X3))

    X_ID, X_geom, X_data = conflict.split_conflict_geom_data(X_in)

    X_out = np.column_stack((X_ID, X_geom, X_data))

    X_false = np.where(np.equal(X_in, X_out) == False)[0]

    assert X_false.size == 0

def test_get_poly_geometry():

    gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    config = create_fake_config()

    list_geometry = conflict.get_poly_geometry(gdf, config)

    assert len(gdf) == len(list_geometry)

def test_get_poly_ID():

    gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    list_ID = conflict.get_poly_ID(gdf)

    assert len(gdf) == len(list_ID)

