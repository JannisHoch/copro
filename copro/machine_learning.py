import os
import pickle
import pandas as pd
import numpy as np
from sklearn import svm, neighbors, ensemble, preprocessing, model_selection, metrics
from copro import migration, data 
from scipy.stats.mstats import winsorize

def define_scaling(config):
    """Defines scaling method based on model configurations.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Raises:
        ValueError: raised if a non-supported scaling method is specified.

    Returns:
        scaler: the specified scaling method instance.
    """

    if config.get('machine_learning', 'scaler') == 'MinMaxScaler':
        scaler = preprocessing.MinMaxScaler()
    elif config.get('machine_learning', 'scaler') == 'StandardScaler':
        scaler = preprocessing.StandardScaler()
    elif config.get('machine_learning', 'scaler') == 'RobustScaler':
        scaler = preprocessing.RobustScaler()
    elif config.get('machine_learning', 'scaler') == 'QuantileTransformer':
        scaler = preprocessing.QuantileTransformer(random_state=0)
    else:
        raise ValueError('no supported scaling-algorithm selected - choose between MinMaxScaler, StandardScaler, RobustScaler or QuantileTransformer')

    if config.getboolean('general', 'verbose'): print('DEBUG: chosen scaling method is {}'.format(scaler))

    return scaler

def define_model(config):
    """Defines model based on model configurations. Model parameter were optimized beforehand using GridSearchCV.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Raises:
        ValueError: raised if a non-supported model is specified.

    Returns:
        model: the specified model instance.
    """    
    
    if config.get('machine_learning', 'model') == 'RFClassifier':
        mdl = ensemble.RandomForestClassifier(n_estimators=1000, class_weight={1: 100}, random_state=0)
    elif config.get('machine_learning', 'model')== 'RFRegression':
        mdl = ensemble.RandomForestRegressor()
    else:
        raise ValueError('no supported ML model selected - choose between RFRegression or RFClassifier')

    if config.getboolean('general', 'verbose'): print('DEBUG: chosen ML model is {}'.format(mdl))

    return mdl

def split_scale_train_test_split(X, Y, config, scaler):
    """Splits and transforms the X-array (or sample data) and Y-array (or target data) in test-data and training-data.
    The fraction of data used to split the data is specified in the configuration file.
    Additionally, the unique identifier and geometry of each data point in both test-data and training-data is retrieved in separate arrays.

    Args:
        X (array): array containing the variable values plus unique identifer and geometry information.
        Y (array): array containing merely the integer migration data.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        scaler (scaler): the specified scaling method instance.

    Returns:
        arrays: arrays containing training-set and test-set for X-data and Y-data as well as IDs and geometry.
    """ 
    ## separate arrays for ID, geometry, and variable values
    X_ID, X_data = migration.split_migration_geom_data(X)  # X_geom

    # scaling only the variable values
    if config.getboolean('general', 'verbose'): print('DEBUG: fitting and transforming X')
    X_ft = scaler.fit_transform(X_data)

    ##- combining ID, geometry and scaled sample values per polygon
    X_cs = np.column_stack((X_ID, X_ft)) # X_geom

    ##- splitting in train and test samples based on user-specified fraction
    if config.getboolean('general', 'verbose'): print('DEBUG: splitting both X and Y in train and test data')
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_cs, Y,
                                                                        test_size=1-config.getfloat('machine_learning', 'train_fraction')) 
    
    if config.get('machine_learning', 'model') == 'RFClassifier':
        y_train = y_train.astype(bool)
        y_test = y_test.astype(bool)
    
    # for training-set and test-set, split in ID, geometry, and values
    X_train_ID, X_train = migration.split_migration_geom_data(X_train)  # X_train_geom, 
    X_test_ID, X_test = migration.split_migration_geom_data(X_test) #X_test_geom, 

    return X_train, X_test, y_train, y_test, X_train_ID, X_test_ID # X_train_geom, X_test_geom 

def fit_predict(X_train, y_train, X_test, mdl, config, out_dir, root_dir, run_nr, migration_gdf):
    """Fits model based on training-data and makes predictions.
    The fitted model is dumped to file with pickle to be used again during projections.
    Makes prediction with test-data including probabilities of those predictions.

    Args:
        X_train (array): training-data of variable values.
        y_train (array): training-data of migration data.
        X_test (array): test-data of variable values.
        mdl: the specified model instance.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        out_dir (path): path to output folder.
        run_nr (int): number of fit/predict repetition and created model.

    Returns:
        arrays: arrays including the predictions made and their probabilities
    """    
        # fit the model with training data  
    if config.getboolean('general', 'weighting_Y_train'): # determine if Y_train should be weighted based on population per polygon 
        gid2_weights = migration.weight_migration(config, root_dir, migration_gdf)
                
        if config.getboolean('migration', 'migration_percentage'):

            # Merge gid2_weights with migration_gdf based on common columns
            merged_data = pd.merge(gid2_weights, migration_gdf[['GID_2', 'year', 'net_migration']], on=['GID_2', 'year'])
            merged_data['migration_ratio'] = merged_data['net_migration'] / merged_data['population_total']

            matching_rows_list = []

            for value in y_train:
                matching_row = merged_data[merged_data['migration_ratio'] == value]
                # Concatenate all the individual DataFrames into a single DataFrame
                matching_rows_list.append(matching_row)
            
            all_matching_rows = pd.concat(matching_rows_list, ignore_index=True) 
            selected_weights = all_matching_rows['weight'].values
            mdl.fit(X_train, y_train, sample_weight=selected_weights)
            print('INFO: ratio Y_test data is weighted')
        
        else:
            matching_rows_list = []
            for value in y_train:
                matching_row = migration_gdf[migration_gdf['net_migration'] == value]
                matching_rows_list.append(matching_row)  

            # Concatenate all the individual DataFrames into a single DataFrame
            all_matching_rows = pd.concat(matching_rows_list, ignore_index=True)  

            # Merge the 'gid2_weights' DataFrame to 'all_matching_rows'
            all_matching_rows = all_matching_rows.merge(gid2_weights, on=['GID_2', 'year'], how='left')
  
            selected_weights = all_matching_rows['weight'].values
            mdl.fit(X_train, y_train, sample_weight=selected_weights)
            print('INFO: Y_train data is weighted')
    
    else: # if no weighing is selected in the cfg-file

        mdl.fit(X_train, y_train)
        print('INFO: Y_train data is not weighted')

    # create folder to store all model with pickle
    mdl_pickle_rep = os.path.join(out_dir, 'mdls')
    if not os.path.isdir(mdl_pickle_rep):
        os.makedirs(mdl_pickle_rep)

    # save the fitted model to file via pickle.dump()
    if config.getboolean('general', 'verbose'): print('DEBUG: dumping model to {}'.format(mdl_pickle_rep))
    with open(os.path.join(mdl_pickle_rep, 'mdl_{}.pkl'.format(run_nr)), 'wb') as f:
        pickle.dump(mdl, f)

    # make prediction
    y_pred = mdl.predict(X_test)

    # make prediction of probability
    if (config.get('machine_learning', 'model')) == 'RFClassifier':
        y_prob = mdl.predict_proba(X_test)
    elif (config.get('machine_learning', 'model')) == 'RFRegression': 
        y_prob = 1 # TEMP fix, is this the right way? mdl.predict_proba(X_test)
   
    return y_pred, y_prob

def pickle_mdl(scaler, mdl, config, root_dir):
    """(Re)fits a model with all available data and pickles it.

    Args:
        scaler (scaler): the specified scaling method instance.
        mdl: the specified model instance.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): path to location of cfg-file.

    Returns:
        model: model fitted with all available data.
    """    

    print('INFO: fitting the model with all data from reference period')

    # reading XY-data
    # if nothing specified in cfg-file, load from output directory
    if config.get('pre_calc', 'XY') is '':
        if config.getboolean('general', 'verbose'): print('DEBUG: loading XY data from {}'.format(os.path.join(root_dir, config.get('general', 'output_dir'), '_REF', 'XY.npy')))
        XY_fit = np.load(os.path.join(root_dir, config.get('general', 'output_dir'), '_REF', 'XY.npy'), allow_pickle=True)
    # if a path is specified, load from there
    else:
        if config.getboolean('general', 'verbose'): print('DEBUG: loading XY data from {}'.format(os.path.join(root_dir, config.get('pre_calc', 'XY'))))
        XY_fit = np.load(os.path.join(root_dir, config.get('pre_calc', 'XY')), allow_pickle=True)

    # split in X and Y data
    X_fit, Y_fit = data.split_XY_data(XY_fit, config)
    # split X in ID, geometry, and values
    X_ID_fit, X_geom_fit, X_data_fit = migration.split_migration_geom_data(X_fit) 
    # scale values
    X_ft_fit = scaler.fit_transform(X_data_fit)
    # fit model with values
    mdl.fit(X_ft_fit, Y_fit)

    return mdl

def load_mdls(config, out_dir):
    """Loads the paths to all previously fitted models to a list.
    Models were saved to file in fit_predict().
    With this list, the models can be loaded again during projections.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        out_dir (path): path to output folder.

    Returns:
        list: list with file names of models.
    """ 

    mdls = os.listdir(os.path.join(out_dir, 'mdls'))

    assert (len(mdls), config.getint('machine_learning', 'n_runs')), AssertionError('ERROR: number of loaded models does not match the specified number of runs in cfg-file!')

    return mdls