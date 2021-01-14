from copro import models, data, machine_learning, evaluation
import pandas as pd
import numpy as np
import os, sys


def create_XY(config, out_dir, root_dir, polygon_gdf, conflict_gdf, projection_period=None):
    """Top-level function to create the X-array and Y-array.
    If the XY-data was pre-computed and specified in cfg-file, the data is loaded.
    If not, variable values and conflict data are read from file and stored in array. The resulting array is by default saved as npy-format to file.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        out_dir (str): path to output folder.
        root_dir (str): path to location of cfg-file.
        polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.
        conflict_gdf (geo-dataframe): geo-dataframe containing the selected conflicts.

    Returns:
        array: X-array containing variable values.
        array: Y-array containing conflict data.
    """    

    if config.get('pre_calc', 'XY') is '':

        XY = data.initiate_XY_data(config)

        XY = data.fill_XY(XY, config, root_dir, conflict_gdf, polygon_gdf, out_dir)

        print('INFO: saving XY data by default to file {}'.format(os.path.join(out_dir, 'XY.npy')))
        np.save(os.path.join(out_dir,'XY'), XY)

    else:

        print('INFO: loading XY data from file {}'.format(os.path.join(root_dir, config.get('pre_calc', 'XY'))))
        XY = np.load(os.path.join(root_dir, config.get('pre_calc', 'XY')), allow_pickle=True)
        
    X, Y = data.split_XY_data(XY, config)    

    return X, Y

def create_X(config, out_dir, root_dir, polygon_gdf, conflict_gdf, projection_period):
    """Top-level function to create the X-array.
    If the X-data was pre-computed and specified in cfg-file, the data is loaded.
    If not, variable values are read from file and stored in array. 
    The resulting array is by default saved as npy-format to file.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        out_dir (str): path to output folder.
        root_dir (str): path to location of cfg-file.
        polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.
        conflict_gdf (geo-dataframe): geo-dataframe containing the selected conflicts.

    Returns:
        array: X-array containing variable values.
    """    

    if config.get('pre_calc', 'XY') is '':

        X = data.initiate_X_data(config)

        X = data.fill_XY(X, config, root_dir, conflict_gdf, polygon_gdf, out_dir, proj=True, proj_period=projection_period)

        print('INFO: saving X data by default to file {}'.format(os.path.join(out_dir, 'X.npy')))
        np.save(os.path.join(out_dir,'X'), X)

    else:

        print('INFO: loading XY data from file {}'.format(os.path.join(root_dir, config.get('pre_calc', 'X'))))
        X = np.load(os.path.join(root_dir, config.get('pre_calc', 'X')), allow_pickle=True)

    return X

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

def run_reference(X, Y, config, scaler, clf, out_dir, run_nr=None):
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
        X_df, y_df, eval_dict = models.all_data(X, Y, config, scaler, clf, out_dir, run_nr=run_nr)
    elif config.getint('general', 'model') == 2:
        X_df, y_df, eval_dict = models.leave_one_out(X, Y, config, scaler, clf, out_dir)
    elif config.getint('general', 'model') == 3:
        X_df, y_df, eval_dict = models.single_variables(X, Y, config, scaler, clf, out_dir)
    elif config.getint('general', 'model') == 4:
        X_df, y_df, eval_dict = models.dubbelsteen(X, Y, config, scaler, clf, out_dir)
    else:
        raise ValueError('the specified model type in the cfg-file is invalid - specify either 1, 2, 3 or 4.')

    return X_df, y_df, eval_dict

def run_prediction(scaler, main_dict, root_dir, selected_polygons_gdf, conflict_gdf):
    """Top-level function to run a predictive model with a already fitted classifier and new data.

    Args:
        X (array): X-array containing variable values.
        scaler (scaler): the specified scaler instance.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): path to location of cfg-file.

    Raises:
        ValueError: raised if another model type than the one using all data is specified in cfg-file.

    Returns:
        datatrame: containing model output on polygon-basis.
    """    

    config_REF = main_dict['_REF'][0]
    out_dir_REF = main_dict['_REF'][1]

    if config_REF.getint('general', 'model') != 1:
        raise ValueError('ERROR: making a prediction is only possible with model type 1, i.e. using all data')

    # initiate output dataframe
    all_y_df = pd.DataFrame(columns=['ID', 'geometry', 'y_pred'])

    # going through each projection specified
    for (each_key, each_val) in config_REF.items('PROJ_files'):

        # get config-object and out-dir per projection
        print('INFO: loading config-object for projection run: {}'.format(each_key))
        config_PROJ = main_dict[str(each_key)][0][0]
        out_dir_PROJ = main_dict[str(each_key)][1]
        print('DEBUG: storing output for this projections to folder {}'.format(out_dir_PROJ))

        # get projection period for this projection
        # defined as all years starting from end of reference run until specified end of projections
        projection_period = models.determine_projection_period(config_REF, config_PROJ, out_dir_PROJ)

        # read sample data for each year
        # i.e. we here start with time stepping
        print('INFO: reading sample data from files')
        X = create_X(config_PROJ, out_dir_PROJ, root_dir, selected_polygons_gdf, conflict_gdf, projection_period=projection_period)

        # put all the data into the machine learning algo
        # here the data will be used to make projections with various classifiers
        y_df = models.predictive(X, scaler, main_dict, root_dir)

        # create one major output dataframe containing all output for all projections with all classifiers
        all_y_df = all_y_df.append(y_df, ignore_index=True)

    return all_y_df