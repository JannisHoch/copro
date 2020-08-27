import os
import pandas as pd
import numpy as np
from sklearn import svm, neighbors, ensemble, preprocessing, model_selection

def define_scaling(config):
    """[summary]

    Args:
        config ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """    

    if config.getboolean('general', 'sensitivity_analysis'):
        scalers = [preprocessing.MinMaxScaler(),
                   preprocessing.StandardScaler(),
                   preprocessing.RobustScaler(),
                   preprocessing.QuantileTransformer(random_state=42)]
    
    elif not config.getboolean('general', 'sensitivity_analysis'):
        if config.get('machine_learning', 'scaler') == 'MinMaxScaler':
            scalers = [preprocessing.MinMaxScaler()]
        elif config.get('machine_learning', 'scaler') == 'StandardScaler':
            scalers = [preprocessing.StandardScaler()]
        elif config.get('machine_learning', 'scaler') == 'RobustScaler':
            scalers = [preprocessing.RobustScaler()]
        elif config.get('machine_learning', 'scaler') == 'QuantileTransformer':
            scalers = [preprocessing.QuantileTransformer(random_state=42)]
        else:
            raise ValueError('no supported scaling-algorithm selected - choose between MinMaxScaler, StandardScaler, RobustScaler or QuantileTransformer')

    print('chosen scaling method is {}'.format(scalers[0]))

    return scalers

def define_model(config):
    """[summary]

    Args:
        config ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """    
    
    if config.getboolean('general', 'sensitivity_analysis'):
        clfs = [svm.NuSVC(nu=0.1, kernel='rbf', class_weight={1: 100}, random_state=42, probability=True, degree=10, gamma=10),
                neighbors.KNeighborsClassifier(n_neighbors=10, weights='distance'),
                ensemble.RandomForestClassifier(n_estimators=1000, class_weight={1: 100}, random_state=42)]

        print('sensitivity analysis specified - all supported models chosen for run')
    
    elif not config.getboolean('general', 'sensitivity_analysis'):
        if config.get('machine_learning', 'model') == 'NuSVC':
            clfs = [svm.NuSVC(nu=0.1, kernel='rbf', class_weight={1: 100}, random_state=42, probability=True, degree=10, gamma=10)]
        elif config.get('machine_learning', 'model') == 'KNeighborsClassifier':
            clfs = [neighbors.KNeighborsClassifier(n_neighbors=10, weights='distance')]
        elif config.get('machine_learning', 'model') == 'RFClassifier':
            clfs = [ensemble.RandomForestClassifier(n_estimators=1000, class_weight={1: 100}, random_state=42)]
        else:
            raise ValueError('no supported ML model selected - choose between NuSVC, KNeighborsClassifier or RFClassifier')

        print('chosen ML model is {}'.format(clfs[0]))

    return clfs

#TODO: it may make sense to have the entire XY process chain as object-oriented code

def split_scale_train_test_split(X, Y, config, scaler):
    """[summary]

    Args:
        X ([type]): [description]
        Y ([type]): [description]
        config ([type]): [description]
        scaler ([type]): [description]

    Returns:
        [type]: [description]
    """    

    ##- separate arrays for geomety and variable values
    X_geom = X[:, 0]
    X_data = X[: , 1:]

    ##- scaling only the variable values
    X_f = scaler.fit_transform(X_data)

    ##- combining geometry and scaled variable values
    X_s = np.column_stack((X_geom, X_f))

    ##- splitting in train and test samples
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_s,
                                                                        Y,
                                                                        test_size=1-config.getfloat('machine_learning', 'train_fraction'))    

    X_train_geom = X_train[:, 0]
    X_test_geom = X_test[:, 0]
    X_train = X_train[: , 1:]
    X_test = X_test[: , 1:]

    return X_train, X_test, y_train, y_test, X_train_geom, X_test_geom