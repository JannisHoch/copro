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
        geo-dataframe: geo-referenced conflict data.
    """     
    
    # get path to file containing data
    conflict_fo = os.path.join(root_dir, config.get('general', 'input_dir'), config.get('conflict', 'conflict_file'))

    # read file to pandas dataframe
    click.echo('INFO: reading csv file to dataframe {}'.format(conflict_fo))
    df = pd.read_csv(conflict_fo)

    # convert dataframe to geo-dataframe
    if config.getboolean('general', 'verbose'): click.echo('DEBUG: translating to geopandas dataframe')
    gdf = gpd.GeoDataFrame(df,
                          geometry=gpd.points_from_xy(df[longitude], df[latitude]),
                          crs=crs)

    return gdf

def show_versions():
    """click.echos the version numbers by the main python-packages used.
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
        sys.exit('please upgrade geopandas to version 0.7.0, your current version is {}. To avoid the problem, make sure CoPro is installed in its own conda environment.'.format(gpd_version))

    click.echo("Python version: {}".format(os_version))
    click.echo("copro version: {}".format(cm_version))
    click.echo("geopandas version: {}".format(gpd_version))
    click.echo("xarray version: {}".format(xr_version))
    click.echo("rasterio version: {}".format(rio_version))
    click.echo("pandas version: {}".format(pd_version))
    click.echo("numpy version: {}".format(np_version))
    click.echo("scikit-learn version: {}".format(skl_version))
    click.echo("matplotlib version: {}".format(mpl_version))
    click.echo("seaborn version: {}".format(sbs_version))
    click.echo("rasterstats version: {}".format(rstats_version))

def parse_settings(settings_file):
    """Reads the model configuration file.

    Args:
        settings_file (str): path to settings-file (cfg-file).

    Returns:
        ConfigParser-object: parsed model configuration.
    """    

    config = RawConfigParser(allow_no_value=True, inline_comment_prefixes='#')
    config.optionxform = lambda option: option
    config.read(settings_file)

    return config

def parse_projection_settings(config, root_dir):
    """This function parses the (various) cfg-files for projections.
    These cfg-files need to be specified one by one in the PROJ_files section of the cfg-file for the reference run.
    The function returns then a dictionary with the name of the run and the associated config-object.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model for the reference run.

    Returns:
        dict: dictionary with name and config-object per specified projection run.
    """    

    # initiate output dictionary
    config_dict = dict()

    # first entry is config-object for reference run
    config_dict['_REF'] = config

    # loop through all keys and values in PROJ_files section of reference config-object
    for (each_key, each_val) in config.items('PROJ_files'):

        # for each value (here representing the cfg-files of the projections), get the absolute path
        each_val = os.path.abspath(os.path.join(root_dir, each_val))

        # parse each config-file specified
        if config.getboolean('general', 'verbose'): click.echo('DEBUG: parsing settings from file {}'.format(each_val))
        each_config = parse_settings(each_val)

        # update the output dictionary with key and config-object
        config_dict[each_key] = ([each_config])

    return config_dict

def make_output_dir(config, root_dir, config_dict):
    """Creates the output folder at location specfied in cfg-file, and returns dictionary with config-objects and out-dir per run.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): absolute path to location of configurations-file
        config_dict (dict): dictionary containing config-objects per projection.

    Returns:
        dict: dictionary containing config-objects and output directories for reference run and all projection runs.
    """    

    # get path to main output directory as specified in cfg-file
    out_dir = os.path.join(root_dir, config.get('general','output_dir'))
    click.echo('INFO: saving output to main folder {}'.format(out_dir))

    # initalize list for all out-dirs
    all_out_dirs = list()

    # append the path to the output folder for the reference run
    # note that this is hardcoded, i.e. each output folder will have a sub-folder '_REF'
    all_out_dirs.append(os.path.join(out_dir, '_REF'))

    # for all specified projections, create individual sub-folder under the folder '_PROJ'
    # and append those to list as well
    out_dir_proj = os.path.join(out_dir, '_PROJ')
    for key, i in zip(config_dict, range(len(config_dict))):
        if i > 0:
            all_out_dirs.append(os.path.join(out_dir_proj, str(key)))

    assert (len(all_out_dirs) == len(config_dict)), AssertionError('ERROR: number of output folders and config-objects do not match!')

    # initiate dictionary for config-objects and out-dir per un
    main_dict = dict()

    # for all keys (i.e. run names), assign config-object (i.e. the values) as well as out-dir
    for key, value, i in zip(config_dict.keys(), config_dict.values(), range(len(config_dict))):
        main_dict[key] = [value, all_out_dirs[i]]

    # create all the specified output folders if they do not exist yet
    # if they exist, remove all files there besides the npy-files
    for key, value in main_dict.items():
        
        # get entry corresponding to out-dir
        # value [0] would be the config-object
        d = value[1]

        # check if out-dir exists and if not, create it
        if not os.path.isdir(d):
            click.echo('INFO: creating output-folder {}'.format(d))
            os.makedirs(d)

        # else, remove all files with a few exceptions
        else:
            for root, dirs, files in os.walk(d):
                if (config.getboolean('general', 'verbose')) and (len(files) > 0): 
                    click.echo('DEBUG: remove files in {}'.format(os.path.abspath(root)))
                for fo in files:
                    if (fo =='XY.npy') or (fo == 'X.npy'):
                        if config.getboolean('general', 'verbose'): click.echo('DEBUG: sparing {}'.format(fo))
                        pass
                    else:
                        os.remove(os.path.join(root, fo))
                            
    return main_dict
    
def download_UCDP(config, root_dir):
    """If specfied in cfg-file, the PRIO/UCDP data is directly downloaded and used as model input.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): absolute path to location of configurations-file
    """    

    # define path where downloaded data will be stored
    path = os.path.join(os.path.join(root_dir, config.get('general', 'input_dir')), 'UCDP')
    # create folder if not there yer
    if not os.path.isdir(path):
        os.mkdir(path)
    
    # URL to be downloaded
    url = 'http://ucdp.uu.se/downloads/ged/ged201-csv.zip'

    # define filename of downloaded object
    filename = os.path.join(path, 'ged201-csv.zip')

    click.echo('INFO: no conflict file was specified, hence downloading data from {} to {}'.format(url, filename))

    # save URL to filename
    urllib.request.urlretrieve(url, filename)

    # path to csv-file
    csv_fo = zipfile.ZipFile(filename, 'r').namelist()[0]
    
    # extract all data
    zipfile.ZipFile(filename, 'r').extractall(path=path)
    
    # set path to csv-file in config-object
    path_set = os.path.join(path, csv_fo)
    config['conflict']['conflict_file'] = path_set

    return

def print_model_info():
    """click.echos a header with main model information.
    """    

    click.echo('')
    click.echo(click.style('#### CoPro version {} ####'.format(copro.__version__), fg='yellow'))
    click.echo(click.style('#### For information about the model, please visit https://copro.readthedocs.io/ ####', fg='yellow'))
    click.echo(click.style('#### Copyright (2020-{}): {} ####'.format(date.today().year, copro.__author__), fg='yellow'))
    click.echo(click.style('#### Contact via: {} ####'.format(copro.__email__), fg='yellow'))
    click.echo(click.style('#### The model can be used and shared under the MIT license ####' + os.linesep, fg='yellow'))

    return

def initiate_setup(settings_file, verbose=None):
    """Initiates the model set-up. 
    It parses the cfg-file, creates an output folder, copies the cfg-file to the output folder, and, if specified, downloads conflict data.

    Args:
        settings_file (str): path to settings-file (cfg-file).
        verbose (bool, optional): whether model is verbose or not, e.g. click.echos DEBUG output or not. If None, then the setting in cfg-file counts. Otherwise verbose can be set directly to function which superseded the cfg-file. Defaults to None.

    Returns:
        ConfigParser-object: parsed model configuration.
        out_dir_list: list with paths to output folders; first main output folder, then reference run folder, then (multiple) folders for projection runs.
        root_dir: path to location of cfg-file.
    """  

    # print model info, i.e. author names, license info etc.
    print_model_info()

    # get name of directory where cfg-file is stored
    root_dir = os.path.dirname(os.path.abspath(settings_file))

    # parse cfg-file and get config-object for reference run
    config = parse_settings(settings_file)
    click.echo('INFO: reading model properties from {}'.format(settings_file))

    if verbose != None:
        config.set('general', 'verbose', str(verbose))

    click.echo('INFO: verbose mode on: {}'.format(config.getboolean('general', 'verbose')))

    # get dictionary with all config-objects, also for projection runs
    config_dict = parse_projection_settings(config, root_dir)

    # get dictionary with all config-objects and all out-dirs
    main_dict = make_output_dir(config, root_dir, config_dict)

    # copy cfg-file of reference run to out-dir of reference run
    if config.getboolean('general', 'verbose'): click.echo('DEBUG: copying cfg-file {} to folder {}'.format(os.path.abspath(settings_file), main_dict['_REF'][1]))
    copyfile(os.path.abspath(settings_file), os.path.join(main_dict['_REF'][1], 'copy_of_{}'.format(os.path.basename(settings_file))))

    # if specfied, download UCDP/PRIO data directly
    if config['conflict']['conflict_file'] == 'download':
        download_UCDP(config)

    # if any other model than all_data is specified, set number of runs to 1
    if (config.getint('general', 'model') == 2) or (config.getint('general', 'model') == 3):
        config.set('machine_learning', 'n_runs', str(1))
        click.echo('INFO: changed nr of runs to {}'.format(config.getint('machine_learning', 'n_runs')))

    return main_dict, root_dir

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

    # stack identifier and geometry of all polygons
    # test if gdf has column 'name', otherwise use column 'watprovID'
    arr = np.column_stack((gdf.watprovID.to_numpy(), gdf.geometry.to_numpy()))

    # convert to dataframe
    df = pd.DataFrame(data=arr, columns=['ID', 'geometry'])

    # use column ID as index
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

    # concatenate dataframes of sample data and target values
    df = pd.concat([X_df, y_df], axis=1)
    # keep only those entries where conflict was observed
    df = df.loc[df.y_test==1]

    # split again into X and Y
    X1_df = df[df.columns[:len(X_df.columns)]]
    y1_df = df[df.columns[len(X_df.columns):]]

    return X1_df, y1_df

def save_to_csv(arg, out_dir, fname):
    """Saves an dictionary to csv-file.

    Args:
        arg (dict): dictionary or dataframe to be saved.
        out_dir (str): path to output folder.
        fname (str): name of stored item.
    """    

    # check if arg is actuall a dict
    if isinstance(arg, dict):
        # create dataframe from dict
        try:
            arg = pd.DataFrame().from_dict(arg)
        except:
            arg = pd.DataFrame().from_dict(arg, orient='index')

    # save dataframe as csv
    arg.to_csv(os.path.join(out_dir, fname + '.csv'))

    return

def save_to_npy(arg, out_dir, fname):
    """Saves an argument (either dictionary or dataframe) to npy-file.

    Args:
        arg (dict or dataframe): dictionary or dataframe to be saved.
        out_dir (str): path to output folder.
        fname (str): name of stored item.
    """    

    # if arg is dict, then first create dataframe, then np-array
    if isinstance(arg, dict):
        arg = pd.DataFrame().from_dict(arg)
        arg = arg.to_numpy()

    # if arg is dataframe, directly create np-array
    elif isinstance(arg, pd.DataFrame):
        arg = arg.to_numpy()

    # save np-array as npy-file
    np.save(os.path.join(out_dir, fname + '.npy'), arg)

    return

def determine_projection_period(config_REF, config_PROJ):
    """Determines the period for which projections need to be made. 
    This is defined as the period between the end year of the reference run and the specified projection year for each projection.

    Args:
        config_REF (ConfigParser-object): object containing the parsed configuration-settings of the model for the reference run.
        config_PROJ (ConfigParser-object): object containing the parsed configuration-settings of the model for a projection run..

    Returns:
        list: list containing all years of the projection period.
    """    

    # get all years of projection period
    projection_period = np.arange(config_REF.getint('settings', 'y_end')+1, config_PROJ.getint('settings', 'y_proj')+1, 1)
    # convert to list
    projection_period = projection_period.tolist()
    print('INFO: the projection period is {} to {}'.format(projection_period[0], projection_period[-1]))

    return projection_period

