from copro import machine_learning, migration, utils, evaluation, data
import pandas as pd
import numpy as np
import pickle
import os, sys

def all_data(X, Y, config, scaler, mdl, out_dir, root_dir, run_nr, migration_gdf):
    """Main model workflow when all XY-data is used. 
    The model workflow is executed for each model.

    Args:
        X (array): array containing the variable values plus IDs and geometry information.
        Y (array): array containing the migration data.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        scaler (scaler): the specified scaling method instance.
        mdl (model): the specified model instance.
        out_dir (str): path to output folder.
        root_dir (str): absolute path to location of configurations-file

    Returns:
        dataframe: containing the test-data X-array values.
        datatrame: containing model output on polygon-basis.
        dict: dictionary containing evaluation metrics per simulation.
    """    
    if config.getboolean('general', 'verbose'): print('DEBUG: using all data')

    # split X into training-set and test-set, scale training-set data
    X_train, X_test, y_train, y_test, X_train_geom, X_test_geom, X_train_ID, X_test_ID = machine_learning.split_scale_train_test_split(X, Y, config, scaler) 

    # convert to dataframe
    X_df = pd.DataFrame(X_test)

    # fit model and make prediction with test-set depending on model choice
    if config.get('machine_learning', 'model') == 'NuSVC' or ('machine_learning', 'model') == 'KNeighborsClassifier' or ('machine_learning', 'model') == 'RFClassifier':
        y_pred, y_prob = machine_learning.fit_predict(X_train, y_train, X_test, mdl, config, out_dir, root_dir, run_nr, migration_gdf)
        y_prob_0 = y_prob[:, 0] # probability to predict 0
        y_prob_1 = y_prob[:, 1] # probability to predict 1 
        # evaluate prediction and save to dict  
        eval_dict = evaluation.evaluate_prediction_classifier(y_test, y_pred, y_prob, X_test, mdl, config) 
        # aggregate predictions per polygon
        y_df = migration.get_pred_migration_geometry_classifier(X_test_ID, y_test, y_pred, y_prob_0, y_prob_1) # deleted X_test_geom, 
    
    elif config.get('machine_learning', 'model') == 'RFRegression':
        y_pred, y_prob = machine_learning.fit_predict(X_train, y_train, X_test, mdl, config, out_dir, root_dir, run_nr, migration_gdf) 
        # evaluate prediction and save to dict
        eval_dict = evaluation.evaluate_prediction_regression(y_test, y_pred, X_test, mdl, config)
        # aggregate predictions per polygon
        y_df = migration.get_pred_migration_geometry_regression(X_test_ID, y_test, y_pred)    # deleted X_test_geom,      

    return X_df, y_df, eval_dict

def predictive(X, mdl, scaler, config):
    """Predictive model to use the already fitted model to make annual projections for the projection period.
    As other models, it reads data which are then scaled and used in conjuction with the model to project net migration.

    Args:
        X (array): array containing the variable values plus unique identifer and geometry information.
        mdl (model): the fitted specified model instance.
        scaler (scaler): the fitted specified scaling method instance.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model for a projection run.

    Returns:
        datatrame: containing model output on polygon-basis.
    """    
     # Transpose the DataFrame - I dont see where this is now going wrong, but if i dont do this the x-projection data is not read properly
    X = X.transpose()
    X.reset_index(inplace=True)
    X.rename(columns={'index': 'poly_ID'}, inplace=True)
    
    # splitting the data from the ID and geometry part of X
    X_ID, X_geom, X_data = migration.split_migration_geom_data(X.to_numpy()) 
    
    num_features = X_data.shape[1]
    print("Number of features in X_data:", num_features)

    # transforming the data
    # fitting is not needed as already happend before
    if config.getboolean('general', 'verbose'): print('DEBUG: transforming the data from projection period')
    
    X_ft = scaler.transform(X_data) 

    # make projection with transformed data
    if config.getboolean('general', 'verbose'): print('DEBUG: making the projections')    
    y_pred = mdl.predict(X_ft)

    # stack together ID, gemoetry, and projection per polygon, and convert to dataframe
    arr = np.column_stack((X_ID, X_geom, y_pred)) 
    y_df = pd.DataFrame(arr, columns=['ID', 'geometry', 'y_pred']) # (Deleted y_prob_0, y_prob_1) maybe also deleted 'geometry' 
    
    return y_df