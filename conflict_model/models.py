from conflict_model import machine_learning, conflict, utils, evaluation
import pandas as pd
import numpy as np

import os, sys

def all_data(X, Y, config, scaler, clf, out_dir):
    """Main model workflow when all data is used. The model workflow is executed for each model simulation.

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

    if not config.getboolean('general', 'verbose'):
        orig_stdout = sys.stdout
        f = open(os.path.join(out_dir, 'out.txt'), 'w')
        sys.stdout = f

    print('### USING ALL DATA ###' + os.linesep)

    X_train, X_test, y_train, y_test, X_train_geom, X_test_geom, X_train_ID, X_test_ID = machine_learning.split_scale_train_test_split(X, Y, config, scaler)
    
    y_pred, y_prob = machine_learning.fit_predict(X_train, y_train, X_test, clf)

    eval_dict = evaluation.evaluate_prediction(y_test, y_pred, y_prob, X_test, clf, config)

    y_df = conflict.get_pred_conflict_geometry(X_test_ID, X_test_geom, y_test, y_pred)

    X_df = pd.DataFrame(X_test)

    if not config.getboolean('general', 'verbose'):
        sys.stdout = orig_stdout
        f.close() 

    return X_df, y_df, eval_dict

def leave_one_out(X, Y, config, scaler, clf, out_dir):
    """Model workflow when each variable is left out from analysis once.
    Not tested yet for more than one simulation!

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

    if not config.getboolean('general', 'verbose'):
        orig_stdout = sys.stdout
        f = open(os.path.join(out_dir, 'out.txt'), 'w')
        sys.stdout = f

    print('### LEAVE ONE OUT MODEL ###' + os.linesep)

    X_train, X_test, y_train, y_test, X_train_geom, X_test_geom, X_train_ID, X_test_ID = machine_learning.split_scale_train_test_split(X, Y, config, scaler)

    for i, key in zip(range(X_train.shape[1]), config.items('env_vars')):

        print('--- removing data for variable {} ---'.format(key[0]) + os.linesep)

        X_train_loo = np.delete(X_train, i, axis=1)
        X_test_loo = np.delete(X_test, i, axis=1)

        sub_out_dir = os.path.join(out_dir, '_only_'+str(key[0]))
        if not os.path.isdir(sub_out_dir):
            os.makedirs(sub_out_dir)

        y_pred, y_prob = machine_learning.fit_predict(X_train_loo, y_train, X_test_loo, clf)

        eval_dict = evaluation.evaluate_prediction(y_test, y_pred, y_prob, X_test_loo, clf, config)

        utils.save_to_csv(eval_dict, sub_out_dir, 'evaluation_metrics')
    
    if not config.getboolean('general', 'verbose'):
        sys.stdout = orig_stdout
        f.close()
    
    sys.exit('With LEAVE-ONE-OUT model, execution stops here.')

def single_variables(X, Y, config, scaler, clf, out_dir):
    """Model workflow when the model is based on only one single variable.
    Not tested yet for more than one simulation!

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

    if not config.getboolean('general', 'verbose'):
        orig_stdout = sys.stdout
        f = open(os.path.join(out_dir, 'out.txt'), 'w')
        sys.stdout = f

    print('### SINGLE VARIABLE MODEL ###' + os.linesep)

    X_train, X_test, y_train, y_test, X_train_geom, X_test_geom, X_train_ID, X_test_ID = machine_learning.split_scale_train_test_split(X, Y, config, scaler)

    for i, key in zip(range(X_train.shape[1]), config.items('env_vars')):

        print('--- single variable model with variable {} ---'.format(key[0]) + os.linesep)

        X_train_svmod = X_train[:, i].reshape(-1, 1)
        X_test_svmod = X_test[:, i].reshape(-1, 1)

        sub_out_dir = os.path.join(out_dir, '_excl_'+str(key[0]))
        if not os.path.isdir(sub_out_dir):
            os.makedirs(sub_out_dir)

        y_pred, y_prob = machine_learning.fit_predict(X_train_svmod, y_train, X_test_svmod, clf)

        eval_dict = evaluation.evaluate_prediction(y_test, y_pred, y_prob, X_test_svmod, clf, config)

        utils.save_to_csv(eval_dict, sub_out_dir, 'evaluation_metrics')

    if not config.getboolean('general', 'verbose'):
        sys.stdout = orig_stdout
        f.close()

    sys.exit('With SINGLE VARIABLE model, execution stops here.')

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

    if not config.getboolean('general', 'verbose'):
        orig_stdout = sys.stdout
        f = open(os.path.join(out_dir, 'out.txt'), 'w')
        sys.stdout = f

    print('### DUBBELSTEENMODEL ###' + os.linesep)

    Y = utils.create_artificial_Y(Y)

    X_train, X_test, y_train, y_test, X_train_geom, X_test_geom, X_train_ID, X_test_ID = machine_learning.split_scale_train_test_split(X, Y, config, scaler)

    y_pred, y_prob = machine_learning.fit_predict(X_train, y_train, X_test, clf)

    eval_dict = evaluation.evaluate_prediction(y_test, y_pred, y_prob, X_test, clf, config)

    y_df = conflict.get_pred_conflict_geometry(X_test_ID, X_test_geom, y_test, y_pred)

    X_df = pd.DataFrame(X_test)
    
    if not config.getboolean('general', 'verbose'):
        sys.stdout = orig_stdout
        f.close()

    return X_df, y_df, eval_dict