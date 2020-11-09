from copro import machine_learning, conflict, utils, evaluation, data
import pandas as pd
import numpy as np
import pickle
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

    print('INFO: using all data')

    X_train, X_test, y_train, y_test, X_train_geom, X_test_geom, X_train_ID, X_test_ID = machine_learning.split_scale_train_test_split(X, Y, config, scaler)
    
    y_pred, y_prob = machine_learning.fit_predict(X_train, y_train, X_test, clf, config)

    eval_dict = evaluation.evaluate_prediction(y_test, y_pred, y_prob, X_test, clf, config)

    y_df = conflict.get_pred_conflict_geometry(X_test_ID, X_test_geom, y_test, y_pred)

    X_df = pd.DataFrame(X_test)

    if not config.getboolean('general', 'verbose'):
        sys.stdout = orig_stdout
        f.close() 

    return X_df, y_df, eval_dict

def leave_one_out(X, Y, config, scaler, clf, out_dir):
    """Model workflow when each variable is left out from analysis once. Output is limited to the metric scores. 
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

    if not config.getboolean('general', 'verbose'):
        orig_stdout = sys.stdout
        f = open(os.path.join(out_dir, 'out.txt'), 'w')
        sys.stdout = f

    raise DeprecationWarning('WARNING: the leave-one-out model will be most likely be deprecated in near future')

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
    
    if not config.getboolean('general', 'verbose'):
        sys.stdout = orig_stdout
        f.close()
    
    sys.exit('INFO: leave-one-out model execution stops here.')

def single_variables(X, Y, config, scaler, clf, out_dir):
    """Model workflow when the model is based on only one single variable. Output is limited to the metric scores. 
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

    if not config.getboolean('general', 'verbose'):
        orig_stdout = sys.stdout
        f = open(os.path.join(out_dir, 'out.txt'), 'w')
        sys.stdout = f

    raise DeprecationWarning('WARNING: the single-variable model will be most likely be deprecated in near future')

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

    if not config.getboolean('general', 'verbose'):
        sys.stdout = orig_stdout
        f.close()

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

    if not config.getboolean('general', 'verbose'):
        orig_stdout = sys.stdout
        f = open(os.path.join(out_dir, 'out.txt'), 'w')
        sys.stdout = f

    print('INFO: dubbelsteenmodel running')

    Y = utils.create_artificial_Y(Y)

    X_train, X_test, y_train, y_test, X_train_geom, X_test_geom, X_train_ID, X_test_ID = machine_learning.split_scale_train_test_split(X, Y, config, scaler)

    y_pred, y_prob = machine_learning.fit_predict(X_train, y_train, X_test, clf, config)

    eval_dict = evaluation.evaluate_prediction(y_test, y_pred, y_prob, X_test, clf, config)

    y_df = conflict.get_pred_conflict_geometry(X_test_ID, X_test_geom, y_test, y_pred)

    X_df = pd.DataFrame(X_test)
    
    if not config.getboolean('general', 'verbose'):
        sys.stdout = orig_stdout
        f.close()

    return X_df, y_df, eval_dict

def predictive(X, scaler, config):
    """Predictive model to use the already fitted classifier to make projections.
    As other models, it reads data which are then scaled and used in conjuction with the classifier to project conflict risk.

    Args:
        X (array): array containing the variable values plus unique identifer and geometry information.
        scaler (scaler): the specified scaling method instance.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Raises:
        ValueError: raised if path to pickled classifier is incorrect.

    Returns:
        datatrame: containing model output on polygon-basis.
    """    

    print('INFO: scaling the data from projection period')
    X = pd.DataFrame(X)
    if config.getboolean('general', 'verbose'): print('DEBUG: number of data points including missing values: {}'.format(len(X)))
    X = X.dropna()
    if config.getboolean('general', 'verbose'): print('DEBUG: number of data points excluding missing values: {}'.format(len(X)))
    X_ID, X_geom, X_data = conflict.split_conflict_geom_data(X.to_numpy())
    ##- scaling only the variable values
    X_ft = scaler.fit_transform(X_data)

    if os.path.isfile(os.path.abspath(config.get('pre_calc', 'clf'))):
        with open(os.path.abspath(config.get('pre_calc', 'clf')), 'rb') as f:
            print('INFO: loading classifier from {}'.format(os.path.abspath(config.get('pre_calc', 'clf'))))
            clf = pickle.load(f)
    else:
        raise ValueError('ERROR: no pre-computed classifier specified in cfg-file, currently specified file {} does not exist'.format(os.path.abspath(config.get('pre_calc', 'clf'))))
        
    print('INFO: making the projection')
    y_pred = clf.predict(X_ft)
    arr = np.column_stack((X_ID, X_geom, y_pred))
    y_df = pd.DataFrame(arr, columns=['ID', 'geometry', 'y_pred'])

    return y_df