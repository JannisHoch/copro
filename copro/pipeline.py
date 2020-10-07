from copro import models, data, machine_learning, evaluation
import pandas as pd
import numpy as np
import os, sys


def create_XY(config, conflict_gdf, polygon_gdf):
    """Top-level function to create the X-array and Y-array.
    If the XY-data was pre-computed and specified in cfg-file, the data is loaded.
    If not, variable values and conflict data are read from file and stored in array. The resulting array is by default saved as npy-format to file.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        conflict_gdf (geo-dataframe): geo-dataframe containing the selected conflicts.
        polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.

    Returns:
        array: X-array containing variable values.
        array: Y-array containing conflict data.
    """    

    if config.get('pre_calc', 'XY') is '':

        XY = data.initiate_XY_data(config)

        XY = data.fill_XY(XY, config, conflict_gdf, polygon_gdf)

        if config.getboolean('general', 'verbose'): 
            print('saving XY data by default to file {}'.format(os.path.abspath(os.path.join(config.get('general', 'input_dir'), 'XY.npy'))) + os.linesep)
        np.save(os.path.join(config.get('general', 'input_dir'),'XY'), XY)

    else:

        if config.getboolean('general', 'verbose'): 
            print('loading XY data from file {}'.format(os.path.abspath(os.path.join(config.get('general', 'input_dir'), config.get('pre_calc', 'XY')))) + os.linesep)
        XY = np.load(os.path.join(config.get('general', 'input_dir'), config.get('pre_calc', 'XY')), allow_pickle=True)
        
    X, Y = data.split_XY_data(XY, config)    

    return X, Y

def prepare_ML(config):
    """Top-level function to instantiate the scaler and model as specified in model configurations.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        scaler: the specified scaler instance.
        classifier: the specified model instance.
    """    

    scaler = machine_learning.define_scaling(config)

    clf = machine_learning.define_model(config)

    return scaler, clf

def run(X, Y, config, scaler, clf, out_dir):
    """Top-level function to run one of the four supported models.

    Args:
        X (array): X-array containing variable values.
        Y (array): Y-array containing conflict data.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        scaler (scaler): the specified scaler instance.
        clf (classifier): the specified model instance.
        out_dir (str): path to output folder.

    Raises:
        ValueError: raised if unsupported model is specified.

    Returns:
        dataframe: containing the test-data X-array values.
        datatrame: containing model output on polygon-basis.
        dict: dictionary containing evaluation metrics per simulation.
    """    

    if config.getint('general', 'model') == 1:
        X_df, y_df, eval_dict = models.all_data(X, Y, config, scaler, clf, out_dir)
    elif config.getint('general', 'model') == 2:
        X_df, y_df, eval_dict = models.leave_one_out(X, Y, config, scaler, clf, out_dir)
    elif config.getint('general', 'model') == 3:
        X_df, y_df, eval_dict = models.single_variables(X, Y, config, scaler, clf, out_dir)
    elif config.getint('general', 'model') == 4:
        X_df, y_df, eval_dict = models.dubbelsteen(X, Y, config, scaler, clf, out_dir)
    else:
        raise ValueError('the specified model type in the cfg-file is invalid - specify either 1, 2, 3 or 4.')

    return X_df, y_df, eval_dict