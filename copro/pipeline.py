from copro import models, data, machine_learning, evaluation, utils
import pandas as pd
import numpy as np
import pickle
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

def create_X(config, out_dir, root_dir, polygon_gdf, conflict_data, proj_year):
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
    X = data.initiate_X_data(config)

    X = data.fill_XY(X, config, root_dir, conflict_data, polygon_gdf, out_dir, proj=True, proj_year=proj_year)

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

def run_prediction(scaler, main_dict, root_dir, selected_polygons_gdf, conflict_data):
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

    clfs = machine_learning.load_clfs(config_REF, out_dir_REF)

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

        if not os.path.isdir(os.path.join(out_dir_PROJ, 'files')):
            os.makedirs(os.path.join(out_dir_PROJ, 'files'))
        if not os.path.isdir(os.path.join(out_dir_PROJ, 'clfs')):
            os.makedirs(os.path.join(out_dir_PROJ, 'clfs'))

        # get projection period for this projection
        # defined as all years starting from end of reference run until specified end of projections
        projection_period = models.determine_projection_period(config_REF, config_PROJ, out_dir_PROJ)

        # for this projection, go through all years
        for i in range(len(projection_period)):

            proj_year = projection_period[i]
            print('INFO: making projection for year {}'.format(proj_year))

            X = data.initiate_X_data(config_PROJ)
            print('INFO: reading sample data from files')
            X = data.fill_X_sample(X, config_PROJ, root_dir, selected_polygons_gdf, proj_year)

            # for the first projection year, we need to fall back on the observed conflict at the last time step of the reference run
            if i == 0:
                print('INFO: reading previous conflicts from file {}'.format(os.path.join(out_dir_REF, 'files', 'conflicts_in_{}.csv'.format(config_REF.getint('settings', 'y_end')))))
                conflict_data = pd.read_csv(os.path.join(out_dir_REF, 'files', 'conflicts_in_{}.csv'.format(config_REF.getint('settings', 'y_end'))), index_col=0)

                print('DEBUG: combining sample data with conflict data from previous year')
                X = data.fill_X_conflict(X, config_PROJ, conflict_data, selected_polygons_gdf, proj_year)
                X = pd.DataFrame.from_dict(X).to_numpy()

            # initiating dataframe containing all projections from all classifiers for this timestep
            y_df = pd.DataFrame(columns=['ID', 'geometry', 'y_pred'])

            # now load all classifiers created in the reference run
            for clf in clfs:

                # creating an individual output folder per classifier
                if not os.path.isdir(os.path.join(os.path.join(out_dir_PROJ, 'clfs', str(clf)))):
                    os.makedirs(os.path.join(out_dir_PROJ, 'clfs', str(clf)))
                
                # load the pickled objects
                with open(os.path.join(out_dir_REF, 'clfs', clf), 'rb') as f:
                    print('DEBUG: loading classifier {} from {}'.format(clf, os.path.join(out_dir_REF, 'clfs')))
                    clf_obj = pickle.load(f)

                # for all other projection years than the first one, we need to read projected conflict from the previous projection year
                if i > 0:
                    print('INFO: reading previous conflicts from file {}'.format(os.path.join(out_dir_PROJ, 'clfs', str(clf), 'projection_for_{}.csv'.format(proj_year-1))))
                    conflict_data = pd.read_csv(os.path.join(out_dir_PROJ, 'clfs', str(clf), 'projection_for_{}.csv'.format(proj_year-1)), index_col=0)

                    print('DEBUG: combining sample data with conflict data for {}'.format(clf))
                    X = data.fill_X_conflict(X, config_PROJ, conflict_data, selected_polygons_gdf, proj_year)
                    X = pd.DataFrame.from_dict(X).to_numpy()

                X = pd.DataFrame(X)
                if config_REF.getboolean('general', 'verbose'): print('DEBUG: number of data points including missing values: {}'.format(len(X)))
                X = X.dropna()
                if config_REF.getboolean('general', 'verbose'): print('DEBUG: number of data points excluding missing values: {}'.format(len(X)))
                
                # put all the data into the machine learning algo
                # here the data will be used to make projections with various classifiers
                # returns the prediction based on one individual classifier
                y_df_clf = models.predictive(X, clf_obj, scaler, main_dict, root_dir)

                # storing the projection per clf to be used in the following timestep
                y_df_clf.to_csv(os.path.join(out_dir_PROJ, 'clfs', str(clf), 'projection_for_{}.csv'.format(proj_year)))

                # removing projection of previous time step as not needed anymore
                if i > 0:
                    os.remove(os.path.join(out_dir_PROJ, 'clfs', str(clf), 'projection_for_{}.csv'.format(proj_year-1)))

                # append to all classifiers dataframe
                y_df = y_df.append(y_df_clf, ignore_index=True)

            # TODO: for testing, stop after 10 years -> needs to be removed!
            if i == 10:
                break

            # y_df.to_csv(os.path.join(out_dir_PROJ, 'clfs', 'all_projections_for_{}.csv'.format(proj_year))) # no need to store this to file

            global_df = utils.global_ID_geom_info(selected_polygons_gdf)

            print('DEBUG: storing model output for year {} to output folder'.format(proj_year))
            df_hit, gdf_hit = evaluation.polygon_model_accuracy(y_df, global_df, out_dir=None, make_proj=True)
            # df_hit = df_hit.drop('geometry', axis=1) # maybe good to keep geometry information if we want to plot it more easily during post-processing
            df_hit.to_csv(os.path.join(out_dir_PROJ, 'output_in_{}.csv'.format(proj_year)))

        # create one major output dataframe containing all output for all projections with all classifiers
        all_y_df = all_y_df.append(y_df, ignore_index=True)

    return all_y_df