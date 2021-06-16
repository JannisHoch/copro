from copro import machine_learning, conflict, utils, evaluation, data
import pandas as pd
import numpy as np
import pickle
import os, sys

def all_data(X, Y, config, scaler, clf, out_dir, run_nr):
    """Main model workflow when all XY-data is used. 
    The model workflow is executed for each classifier.

    Args:
        X (array): array containing the variable values plus IDs and geometry information.
        Y (array): array containing merely the binary conflict classifier data.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        scaler (scaler): the specified scaling method instance.
        clf (classifier): the specified model instance.
        out_dir (str): path to output folder.

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

    # fit classifier and make prediction with test-set
    y_pred, y_prob = machine_learning.fit_predict(X_train, y_train, X_test, clf, config, out_dir, run_nr)
    y_prob_0 = y_prob[:, 0] # probability to predict 0
    y_prob_1 = y_prob[:, 1] # probability to predict 1

    # evaluate prediction and save to dict
    eval_dict = evaluation.evaluate_prediction(y_test, y_pred, y_prob, X_test, clf, config)

    # aggregate predictions per polygon
    y_df = conflict.get_pred_conflict_geometry(X_test_ID, X_test_geom, y_test, y_pred, y_prob_0, y_prob_1)

    return X_df, y_df, eval_dict

def leave_one_out(X, Y, config, scaler, clf, out_dir):
    """Model workflow when each variable is left out from analysis once. 
    Output is limited to the metric scores. 
    Output is stored to sub-folders of the output directory, with each sub-folder containing output for a n-1 variable combination.
    After computing metric scores per prediction (i.e. n-1 variables combinations), model exit is forced.
    Not tested yet for more than one simulation!

    Args:
        X (array): array containing the variable values plus unique identifer and geometry information.
        Y (array): array containing merely the binary conflict classifier data.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        scaler (scaler): the specified scaling method instance.
        clf (classifier): the specified model instance.
        out_dir (str): path to output folder.

    Raises:
        DeprecationWarning: this function will most likely be deprecated due to lack of added value and applicability.
    """    

    raise DeprecationWarning('WARNING: the leave-one-out model is not supported anymore and will be deprecated in a future release')

    X_train, X_test, y_train, y_test, X_train_geom, X_test_geom, X_train_ID, X_test_ID = machine_learning.split_scale_train_test_split(X, Y, config, scaler)

    for i, key in zip(range(X_train.shape[1]), config.items('data')):

        print('INFO: removing data for variable {}'.format(key[0]))

        X_train_loo = np.delete(X_train, i, axis=1)
        X_test_loo = np.delete(X_test, i, axis=1)

        sub_out_dir = os.path.join(out_dir, '_only_'+str(key[0]))
        if not os.path.isdir(sub_out_dir):
            os.makedirs(sub_out_dir)

        y_pred, y_prob = machine_learning.fit_predict(X_train_loo, y_train, X_test_loo, clf, config)

        eval_dict = evaluation.evaluate_prediction(y_test, y_pred, y_prob, X_test_loo, clf, config)

        utils.save_to_csv(eval_dict, sub_out_dir, 'evaluation_metrics')
    
    sys.exit('INFO: leave-one-out model execution stops here.')

def single_variables(X, Y, config, scaler, clf, out_dir):
    """Model workflow when the model is based on only one single variable. 
    Output is limited to the metric scores. 
    Output is stored to sub-folders of the output directory, with each sub-folder containing output for a 1 variable combination.
    After computing metric scores per prediction (i.e. per variable), model exit is forced.
    Not tested yet for more than one simulation!

    Args:
        X (array): array containing the variable values plus unique identifer and geometry information.
        Y (array): array containing merely the binary conflict classifier data.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        scaler (scaler): the specified scaling method instance.
        clf (classifier): the specified model instance.
        out_dir (str): path to output folder.

    Raises:
        DeprecationWarning: this function will most likely be deprecated due to lack of added value and applicability.
    """    

    raise DeprecationWarning('WARNING: the single-variable model is not supported anymore and will be deprecated in a future release')

    X_train, X_test, y_train, y_test, X_train_geom, X_test_geom, X_train_ID, X_test_ID = machine_learning.split_scale_train_test_split(X, Y, config, scaler)

    for i, key in zip(range(X_train.shape[1]), config.items('data')):

        print('INFO: single-variable model with variable {}'.format(key[0]))

        X_train_svmod = X_train[:, i].reshape(-1, 1)
        X_test_svmod = X_test[:, i].reshape(-1, 1)

        sub_out_dir = os.path.join(out_dir, '_excl_'+str(key[0]))
        if not os.path.isdir(sub_out_dir):
            os.makedirs(sub_out_dir)

        y_pred, y_prob = machine_learning.fit_predict(X_train_svmod, y_train, X_test_svmod, clf, config)

        eval_dict = evaluation.evaluate_prediction(y_test, y_pred, y_prob, X_test_svmod, clf, config)

        utils.save_to_csv(eval_dict, sub_out_dir, 'evaluation_metrics')

    sys.exit('INFO: single-variable model execution stops here.')

def dubbelsteen(X, Y, config, scaler, clf, out_dir):
    """Model workflow when the relation between variables and conflict is based on randomness.
    Thereby, the fraction of actual conflict is equal to observations, but the location in array is randomized by shuffling.
    The model workflow is executed for each model simulation.

    Args:
        X (array): array containing the variable values plus unique identifer and geometry information.
        Y (array): array containing merely the binary conflict classifier data.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        scaler (scaler): the specified scaling method instance.
        clf (classifier): the specified model instance.
        out_dir (str): path to output folder.

    Returns:
        dataframe: containing the test-data X-array values.
        datatrame: containing model output on polygon-basis.
        dict: dictionary containing evaluation metrics per simulation.
    """   

    print('INFO: dubbelsteenmodel running')
    raise DeprecationWarning('WARNING: the dubbelsteenmodel model is not supported anymore and will be deprecated in a future release')

    Y = utils.create_artificial_Y(Y)

    X_train, X_test, y_train, y_test, X_train_geom, X_test_geom, X_train_ID, X_test_ID = machine_learning.split_scale_train_test_split(X, Y, config, scaler)

    y_pred, y_prob = machine_learning.fit_predict(X_train, y_train, X_test, clf, config)

    eval_dict = evaluation.evaluate_prediction(y_test, y_pred, y_prob, X_test, clf, config)

    y_df = conflict.get_pred_conflict_geometry(X_test_ID, X_test_geom, y_test, y_pred)

    X_df = pd.DataFrame(X_test)

    return X_df, y_df, eval_dict

def predictive(X, clf, scaler, config):
    """Predictive model to use the already fitted classifier to make annual projections for the projection period.
    As other models, it reads data which are then scaled and used in conjuction with the classifier to project conflict risk.

    Args:
        X (array): array containing the variable values plus unique identifer and geometry information.
        clf (classifier): the fitted specified classifier instance.
        scaler (scaler): the fitted specified scaling method instance.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model for a projection run.

    Returns:
        datatrame: containing model output on polygon-basis.
    """    

    # splitting the data from the ID and geometry part of X
    X_ID, X_geom, X_data = conflict.split_conflict_geom_data(X.to_numpy())

    # transforming the data
    # fitting is not needed as already happend before
    if config.getboolean('general', 'verbose'): print('DEBUG: transforming the data from projection period')
    X_ft = scaler.transform(X_data)

    # make projection with transformed data
    if config.getboolean('general', 'verbose'): print('DEBUG: making the projections')    
    y_pred = clf.predict(X_ft)

    # predict probabilites of outcomes
    y_prob = clf.predict_proba(X_ft)
    y_prob_0 = y_prob[:, 0] # probability to predict 0
    y_prob_1 = y_prob[:, 1] # probability to predict 1

    # stack together ID, gemoetry, and projection per polygon, and convert to dataframe
    arr = np.column_stack((X_ID, X_geom, y_pred, y_prob_0, y_prob_1))
    y_df = pd.DataFrame(arr, columns=['ID', 'geometry', 'y_pred', 'y_prob_0', 'y_prob_1'])
    
    return y_df