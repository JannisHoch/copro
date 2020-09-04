import geopandas as gpd
import pandas as pd
import numpy as np
import os, sys
import urllib.request
import zipfile
from configparser import RawConfigParser
from shutil import copyfile, rmtree
from sklearn import utils

def get_geodataframe(config, longitude='longitude', latitude='latitude', crs='EPSG:4326'):
    """Georeferences a pandas dataframe using longitude and latitude columns of that dataframe. 

    Arguments:
        config {configuration}: parsed configuration settings

    Keyword Arguments:
        longitude {str}: column name with longitude coordinates (default: 'longitude')
        latitude {str}: column name with latitude coordinates (default: 'latitude')
        crs {str}: coordinate system to be used for georeferencing (default: 'EPSG:4326')

    Returns:
        gdf {geodataframe}: geodataframe containing entries with conflicts
    """    

    # construct path to file with conflict data
    conflict_fo = os.path.join(os.path.abspath(config.get('general', 'input_dir')), 
                               config.get('conflict', 'conflict_file'))

    # read file to pandas dataframe
    if config.getboolean('general', 'verbose'): print('reading csv file to dataframe {}'.format(conflict_fo) + os.linesep)
    df = pd.read_csv(conflict_fo)

    if config.getboolean('general', 'verbose'): print('translating to geopandas dataframe' + os.linesep)
    gdf = gpd.GeoDataFrame(df,
                          geometry=gpd.points_from_xy(df[longitude], df[latitude]),
                          crs=crs)

    return gdf

def show_versions():
    """[summary]
    """ 
       
    from conflict_model import __version__ as cm_version
    from geopandas import __version__ as gpd_version
    from pandas import __version__ as pd_version
    from numpy import __version__ as np_version
    from matplotlib import __version__ as mpl_version
    from rasterstats import __version__ as rstats_version
    from xarray import __version__ as xr_version
    from rasterio import __version__ as rio_version
    from sys import version as os_version
    from seaborn import __version__ as sbs_version
    from sklearn import __version__ as skl_version

    #- Geopandas versions lower than 0.7.0 do not yet have the clip function
    if gpd_version < '0.7.0':
        sys.exit('please upgrade geopandas to version 0.7.0, your current version is {}'.format(gpd_version))

    print("Python version: {}".format(os_version))
    print("conflict_model version: {}".format(cm_version))
    print("geopandas version: {}".format(gpd_version))
    print("xarray version: {}".format(xr_version))
    print("rasterio version: {}".format(rio_version))
    print("pandas version: {}".format(pd_version))
    print("numpy version: {}".format(np_version))
    print("scikit-learn version: {}".format(skl_version))
    print("matplotlib version: {}".format(mpl_version))
    print("seaborn version: {}".format(sbs_version))
    print("rasterstats version: {}".format(rstats_version))

def parse_settings(settings_file):

    config = RawConfigParser(allow_no_value=True, inline_comment_prefixes='#')
    config.optionxform = lambda option: option
    config.read(settings_file)

    return config

def make_output_dir(config):

    out_dir = os.path.abspath(config.get('general','output_dir'))
    if os.path.isdir(out_dir):
        rmtree(out_dir)
    os.makedirs(out_dir)
    print('saving output to folder {}'.format(out_dir))

    return out_dir
    
def download_PRIO(config):

    path = os.path.join(os.path.abspath(config.get('general', 'input_dir')), 'UCDP')
    
    url = 'http://ucdp.uu.se/downloads/ged/ged201-csv.zip'

    filename = os.path.join(path, 'ged201-csv.zip')

    print('')
    print('no conflict file was specified, hence downloading data from {} to {}'.format(url, filename) + os.linesep)

    urllib.request.urlretrieve(url, filename)

    csv_fo = zipfile.ZipFile(filename, 'r').namelist()[0]
    
    zipfile.ZipFile(filename, 'r').extractall(path=path)
    
    path_set = os.path.join(path, csv_fo)
    
    config['conflict']['conflict_file'] = path_set

    return

def initiate_setup(settings_file):

    config = parse_settings(settings_file)

    out_dir = make_output_dir(config)

    copyfile(settings_file, os.path.join(out_dir, 'copy_of_run_setting.cfg'))

    if config['conflict']['conflict_file'] == 'download':
        download_PRIO(config)

    return config, out_dir

def create_artificial_Y(Y):

    arr_1 = np.ones(len(np.where(Y != 0)[0]))
    arr_0 = np.zeros(int(len(Y) - len(np.where(Y != 0)[0])))
    Y_r_1 = np.append(arr_1, arr_0)

    assert len(Y), len(Y_r_1)

    Y_r = utils.shuffle(Y_r_1, random_state=42)

    assert len(np.where(Y_r != 0)[0]), len(np.where(Y != 0)[0])

    return Y_r

def global_ID_geom_info(gdf):

    try:
        global_list = np.column_stack((gdf.name.to_numpy(), gdf.geometry.to_numpy()))
    except:
        global_list = np.column_stack((gdf.watprovID.to_numpy(), gdf.geometry.to_numpy()))

    df = pd.DataFrame(data=global_list, columns=['ID', 'geometry'])

    df.set_index(df.ID, inplace=True)
    df = df.drop('ID', axis=1)

    return df

def get_conflict_datapoints_only(X_df, y_df):

    df = pd.concat([X_df, y_df], axis=1)
    df = df.loc[df.y_test==1]

    X1_df = df[df.columns[:len(X_df.columns)]]
    y1_df = df[df.columns[len(X_df.columns):]]

    return X1_df, y1_df

