from conflict_model import machine_learning, conflict, utils, evaluation
import numpy as np

import os, sys

def all_data(X, Y, config, scalers, clfs, out_dir):

    sub_out_dir = os.path.join(out_dir, '_all_data')
    if not os.path.isdir(sub_out_dir):
        os.makedirs(sub_out_dir)

    if not config.getboolean('general', 'verbose'):
        orig_stdout = sys.stdout
        f = open(os.path.join(sub_out_dir, 'out.txt'), 'w')
        sys.stdout = f

    print('### USING ALL DATA ###' + os.linesep)

    if config.getboolean('general', 'sensitivity_analysis'):

        print('--- sensitivy analysis is ON, all combos of scaler and model are applied ---' + os.linesep)

    for scaler in scalers:

        sub_sub_out_dir = os.path.join(sub_out_dir, str(scaler).rsplit('(')[0])
        if not os.path.isdir(sub_sub_out_dir):
            os.makedirs(sub_sub_out_dir)

        X_train, X_test, y_train, y_test, X_train_geom, X_test_geom, X_train_ID, X_test_ID = machine_learning.split_scale_train_test_split(X, Y, config, scaler)

        for clf in clfs:

            sub_sub_sub_out_dir = os.path.join(sub_sub_out_dir, str(clf).rsplit('(')[0])
            if not os.path.isdir(sub_sub_sub_out_dir):
                os.makedirs(sub_sub_sub_out_dir)
            
            y_pred, y_prob = machine_learning.fit_predict(X_train, y_train, X_test, clf)

            evaluation.evaluate_prediction(y_test, y_pred, y_prob, X_test, clf, sub_sub_sub_out_dir)

            y_df, y_gdf = conflict.get_pred_conflict_geometry(X_test_ID, X_test_geom, y_test, y_pred)

    if not config.getboolean('general', 'verbose'):
        sys.stdout = orig_stdout
        f.close() 

    return y_df, y_gdf

def leave_one_out(X, Y, config, scalers, clfs, out_dir):

    if (len(scalers) > 1) or (len(clfs) > 1):
        raise ValueError('not supported with sensitivity analysis model - please select only one scaling and one model technique.')

    sub_out_dir = os.path.join(out_dir, '_leave_one_out_analysis')
    if not os.path.isdir(sub_out_dir):
        os.makedirs(sub_out_dir)

    if not config.getboolean('general', 'verbose'):
        orig_stdout = sys.stdout
        f = open(os.path.join(sub_out_dir, 'out.txt'), 'w')
        sys.stdout = f

    print('### LEAVE ONE OUT MODEL ###' + os.linesep)

    scaler = scalers[0]
    clf = clfs[0]

    X_train, X_test, y_train, y_test, X_train_geom, X_test_geom, X_train_ID, X_test_ID = machine_learning.split_scale_train_test_split(X, Y, config, scaler)

    for i, key in zip(range(X_train.shape[1]), config.items('env_vars')):

        print('--- removing data for variable {} ---'.format(key[0]) + os.linesep)

        X_train_loo = np.delete(X_train, i, axis=1)
        X_test_loo = np.delete(X_test, i, axis=1)

        sub_sub_out_dir = os.path.join(sub_out_dir, '_only_'+str(key[0]))
        if not os.path.isdir(sub_sub_out_dir):
            os.makedirs(sub_sub_out_dir)

        y_pred, y_prob = machine_learning.fit_predict(X_train_loo, y_train, X_test_loo, clf)

        evaluation.evaluate_prediction(y_test, y_pred, y_prob, X_test_loo, clf, sub_sub_out_dir)

        y_df, y_gdf = conflict.get_pred_conflict_geometry(X_test_ID, X_test_geom, y_test, y_pred)

    if not config.getboolean('general', 'verbose'):
        sys.stdout = orig_stdout
        f.close()
    
    return y_df, y_gdf

def single_variables(X, Y, config, scalers, clfs, out_dir):

    if (len(scalers) > 1) or (len(clfs) > 1):
        raise ValueError('not supported with sensitivity analysis model - please select only one scaling and one model technique.')

    sub_out_dir = os.path.join(out_dir, '_single_var_model')
    if not os.path.isdir(sub_out_dir):
        os.makedirs(sub_out_dir)

    if not config.getboolean('general', 'verbose'):
        orig_stdout = sys.stdout
        f = open(os.path.join(sub_out_dir, 'out.txt'), 'w')
        sys.stdout = f

    print('### SINGLE VARIABLE MODEL ###' + os.linesep)

    scaler = scalers[0]
    clf = clfs[0]

    X_train, X_test, y_train, y_test, X_train_geom, X_test_geom, X_train_ID, X_test_ID = machine_learning.split_scale_train_test_split(X, Y, config, scaler)

    for i, key in zip(range(X_train.shape[1]), config.items('env_vars')):

        print('--- single variable model with variable {} ---'.format(key[0]) + os.linesep)

        X_train_svmod = X_train[:, i].reshape(-1, 1)
        X_test_svmod = X_test[:, i].reshape(-1, 1)

        sub_sub_out_dir = os.path.join(sub_out_dir, '_excl_'+str(key[0]))
        if not os.path.isdir(sub_sub_out_dir):
            os.makedirs(sub_sub_out_dir)

        y_pred, y_prob = machine_learning.fit_predict(X_train_svmod, y_train, X_test_svmod, clf)

        evaluation.evaluate_prediction(y_test, y_pred, y_prob, X_test_svmod, clf, sub_sub_out_dir)

        y_df, y_gdf = conflict.get_pred_conflict_geometry(X_test_ID, X_test_geom, y_test, y_pred)

    if not config.getboolean('general', 'verbose'):
        sys.stdout = orig_stdout
        f.close()

    return y_df, y_gdf

def dubbelsteen(X, Y, config, scalers, clfs, out_dir):

    if (len(scalers) > 1) or (len(clfs) > 1):
        raise ValueError('not supported with sensitivity analysis model - please select only one scaling and one model technique.')

    sub_out_dir = os.path.join(out_dir, '_dubbelsteenmodel')
    if not os.path.isdir(sub_out_dir):
        os.makedirs(sub_out_dir)

    if not config.getboolean('general', 'verbose'):
        orig_stdout = sys.stdout
        f = open(os.path.join(sub_out_dir, 'out.txt'), 'w')
        sys.stdout = f

    print('### DUBBELSTEENMODEL ###' + os.linesep)

    scaler = scalers[0]
    clf = clfs[0]

    Y = utils.create_artificial_Y(Y)

    X_train, X_test, y_train, y_test, X_train_geom, X_test_geom, X_train_ID, X_test_ID = machine_learning.split_scale_train_test_split(X, Y, config, scaler)

    y_pred, y_prob = machine_learning.fit_predict(X_train, y_train, X_test, clf)

    evaluation.evaluate_prediction(y_test, y_pred, y_prob, X_test, clf, sub_out_dir)

    y_df, y_gdf = conflict.get_pred_conflict_geometry(X_test_ID, X_test_geom, y_test, y_pred)

    if not config.getboolean('general', 'verbose'):
        sys.stdout = orig_stdout
        f.close()

    return y_df, y_gdf