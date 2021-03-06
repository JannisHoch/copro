import geopandas as gpd
import pandas as pd
import numpy as np
import os, sys
import urllib.request
import zipfile
from configparser import RawConfigParser
from shutil import copyfile
from sklearn import utils
from datetime import date
import click
import copro

def get_geodataframe(config, root_dir, longitude='longitude', latitude='latitude', crs='EPSG:4326'):
    """Georeferences a pandas dataframe using longitude and latitude columns of that dataframe.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): path to location of cfg-file.
        longitude (str, optional): column name with longitude coordinates. Defaults to 'longitude'.
        latitude (str, optional): column name with latitude coordinates. Defaults to 'latitude'.
        crs (str, optional): coordinate system to be used for georeferencing. Defaults to 'EPSG:4326'.

    Returns:
        geo-dataframe: ge-referenced conflict data.
    """     
    
    conflict_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('conflict', 'conflict_file'))

    # read file to pandas dataframe
    print('INFO: reading csv file to dataframe {}'.format(conflict_fo))
    df = pd.read_csv(conflict_fo)

    if config.getboolean('general', 'verbose'): print('DEBUG: translating to geopandas dataframe')
    gdf = gpd.GeoDataFrame(df,
                          geometry=gpd.points_from_xy(df[longitude], df[latitude]),
                          crs=crs)

    return gdf

def show_versions():
    """Prints the version numbers by the main python-packages used.
    """ 
       
    from copro import __version__ as cm_version
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
    print("copro version: {}".format(cm_version))
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
    """Reads the model configuration file.

    Args:
        settings_file (str): path to settings-file (cfg-file).

    Returns:
        - ConfigParser-object: parsed model configuration.
        - str: absolute path to location of configurations-file.
    """    

    config = RawConfigParser(allow_no_value=True, inline_comment_prefixes='#')
    config.optionxform = lambda option: option
    config.read(settings_file)
    root_dir = os.path.dirname(os.path.abspath(settings_file))

    return config, root_dir

def make_output_dir(config, root_dir):
    """Creates the output folder at location specfied in cfg-file.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): absolute path to location of configurations-file

    Returns:
        str: path to output folder
    """    

    out_dir = os.path.join(root_dir, config.get('general','output_dir'))
    print('INFO: saving output to folder {}'.format(out_dir))

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    else:
        for root, dirs, files in os.walk(out_dir):
            if config.getboolean('general', 'verbose'): print('DEBUG: remove files in folder {}'.format(os.path.abspath(root)))
            for fo in files:
                # print(fo)
                if (fo == 'clf.pkl') or (fo =='XY.npy') or (fo == 'X.npy'):
                    if config.getboolean('general', 'verbose'): print('DEBUG: sparing {}'.format(fo))
                    pass
                else:
                    os.remove(os.path.join(root, fo))

    return out_dir
    
def download_PRIO(config, root_dir):
    """If specfied in cfg-file, the PRIO/UCDP data is directly downloaded and used as model input.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): absolute path to location of configurations-file
    """    

    path = os.path.join(os.path.join(root_dir, config.get('general', 'input_dir')), 'UCDP')

    if not os.path.isdir(path):
        os.mkdir(path)
    
    url = 'http://ucdp.uu.se/downloads/ged/ged201-csv.zip'

    filename = os.path.join(path, 'ged201-csv.zip')

    print('INFO: no conflict file was specified, hence downloading data from {} to {}'.format(url, filename))

    urllib.request.urlretrieve(url, filename)

    csv_fo = zipfile.ZipFile(filename, 'r').namelist()[0]
    
    zipfile.ZipFile(filename, 'r').extractall(path=path)
    
    path_set = os.path.join(path, csv_fo)
    
    config['conflict']['conflict_file'] = path_set

    return

def print_model_info():
    """Prints a header with main model information.
    """    

    click.echo('')
    click.echo(click.style('#### CoPro version {} ####'.format(copro.__version__), fg='yellow'))
    click.echo(click.style('#### For information about the model, please visit https://copro.readthedocs.io/ ####', fg='yellow'))
    click.echo(click.style('#### Copyright (2020-{}): {} ####'.format(date.today().year, copro.__author__), fg='yellow'))
    click.echo(click.style('#### Contact via: {} ####'.format(copro.__email__), fg='yellow'))
    click.echo(click.style('#### The model can be used and shared under the MIT license ####' + os.linesep, fg='yellow'))

    return

def initiate_setup(settings_file):
    """Initiates the model set-up. It parses the cfg-file, creates an output folder, copies the cfg-file to the output folder, and, if specified, downloads conflict data.

    Args:
        settings_file (str): path to settings-file (cfg-file).

    Returns:
        ConfigParser-object: parsed model configuration.
        out_dir: path to output folder.
        root_dir: path to location of cfg-file.
    """  

    print_model_info() 

    config, root_dir = parse_settings(settings_file)

    print('INFO: verbose mode on: {}'.format(config.getboolean('general', 'verbose')))

    out_dir = make_output_dir(config, root_dir)

    copyfile(settings_file, os.path.join(out_dir, 'copy_of_run_setting.cfg'))

    if config['conflict']['conflict_file'] == 'download':
        download_PRIO(config)

    if (config.getint('general', 'model') == 2) or (config.getint('general', 'model') == 3):
        config.set('settings', 'n_runs', str(1))
        print('INFOL changed nr of runs to {}'.format(config.getint('settings', 'n_runs')))

    return config, out_dir, root_dir

def create_artificial_Y(Y):
    """Creates an array with identical percentage of conflict points as input array.

    Args:
        Y (array): original array containing binary conflict classifier data.

    Returns:
        array: array with reshuffled conflict classifier data.
    """    

    arr_1 = np.ones(len(np.where(Y != 0)[0]))
    arr_0 = np.zeros(int(len(Y) - len(np.where(Y != 0)[0])))
    Y_r_1 = np.append(arr_1, arr_0)

    Y_r = utils.shuffle(Y_r_1, random_state=42)

    return Y_r

def global_ID_geom_info(gdf):
    """Retrieves unique ID and geometry information from geo-dataframe for a global look-up dataframe. 
    The IDs currently supported are 'name' or 'watprovID'.

    Args:
        gdf (geo-dataframe): containing all polygons used in the model.

    Returns:
        dataframe: look-up dataframe associated ID with geometry
    """    

    try:
        global_list = np.column_stack((gdf.name.to_numpy(), gdf.geometry.to_numpy()))
    except:
        global_list = np.column_stack((gdf.watprovID.to_numpy(), gdf.geometry.to_numpy()))

    df = pd.DataFrame(data=global_list, columns=['ID', 'geometry'])

    df.set_index(df.ID, inplace=True)
    df = df.drop('ID', axis=1)

    return df

def get_conflict_datapoints_only(X_df, y_df):
    """Filters out only those polygons where conflict was actually observed in the test-sample.

    Args:
        X_df (dataframe): variable values per polygon.
        y_df (dataframe): conflict data per polygon.

    Returns:
        dataframe: variable values for polyons where conflict was observed.
        dataframe: conflict data for polyons where conflict was observed.
    """    

    df = pd.concat([X_df, y_df], axis=1)
    df = df.loc[df.y_test==1]

    X1_df = df[df.columns[:len(X_df.columns)]]
    y1_df = df[df.columns[len(X_df.columns):]]

    return X1_df, y1_df

def save_to_csv(arg, out_dir, fname):
    """Saves an argument (either dictionary or dataframe) to csv-file.

    Args:
        arg (dict or dataframe): dictionary or dataframe to be saved.
        out_dir (str): path to output folder.
        fname (str): name of stored item.
    """    

    if isinstance(arg, dict):
        try:
            arg = pd.DataFrame().from_dict(arg)
        except:
            arg = pd.DataFrame().from_dict(arg, orient='index')
    arg.to_csv(os.path.join(out_dir, fname + '.csv'))

    return

def save_to_npy(arg, out_dir, fname):
    """Saves an argument (either dictionary or dataframe) to parquet-file.

    Args:
        arg (dict or dataframe): dictionary or dataframe to be saved.
        out_dir (str): path to output folder.
        fname (str): name of stored item.
    """    

    if isinstance(arg, dict):
        arg = pd.DataFrame().from_dict(arg)
        arg = arg.to_numpy()
    elif isinstance(arg, pd.DataFrame):
        arg = arg.to_numpy()

    np.save(os.path.join(out_dir, fname + '.npy'), arg)

    return

