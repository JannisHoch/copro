import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, neighbors, ensemble, preprocessing, model_selection, metrics
from copro import conflict, data

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
        scaler = preprocessing.QuantileTransformer()
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
        classifier: the specified model instance.
    """    
    
    if config.get('machine_learning', 'model') == 'NuSVC':
        clf = svm.NuSVC(nu=0.1, kernel='rbf', class_weight={1: 100}, probability=True, degree=10, gamma=10)
    elif config.get('machine_learning', 'model') == 'KNeighborsClassifier':
        clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='distance')
    elif config.get('machine_learning', 'model') == 'RFClassifier':
        clf = ensemble.RandomForestClassifier(n_estimators=1000, class_weight={1: 100})
    else:
        raise ValueError('no supported ML model selected - choose between NuSVC, KNeighborsClassifier or RFClassifier')

    if config.getboolean('general', 'verbose'): print('DEBUG: chosen ML model is {}'.format(clf))

    return clf

def split_scale_train_test_split(X, Y, config, scaler):
    """Splits and transforms the X-array and Y-array in test-data and training-data.
    The fraction of data used to split the data is specified in the configuration file.
    Additionally, the unique identifier and geometry of each data point in both test-data and training-data is retrieved in separate arrays.

    Args:
        X (array): array containing the variable values plus unique identifer and geometry information.
        Y (array): array containing merely the binary conflict classifier data.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        scaler (scaler): the specified scaling method instance.

    Raises:
        AssertionError: raised if after all manipulations the number of unique identifiers does not match number of data points in test-data.
        AssertionError: raised if after all manipulations the number of geometries does not match number of data points in test-data.

    Returns:
        arrays: arrays containing training-data and test-data as well as IDs and geometry for training-data and test-data.
    """ 

    ##- separate arrays for geomety and variable values
    X_ID, X_geom, X_data = conflict.split_conflict_geom_data(X)

    if config.getboolean('general', 'verbose'): print('DEBUG: fitting and transforming X')
    ##- scaling only the variable values
    X_ft = scaler.fit_transform(X_data)

    ##- combining geometry and scaled variable values
    X_cs = np.column_stack((X_ID, X_geom, X_ft))

    if config.getboolean('general', 'verbose'): print('DEBUG: splitting both X and Y in train and test data')
    ##- splitting in train and test samples
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_cs,
                                                                        Y,
                                                                        test_size=1-config.getfloat('machine_learning', 'train_fraction'))    

    X_train_ID, X_train_geom, X_train = conflict.split_conflict_geom_data(X_train)
    X_test_ID, X_test_geom, X_test = conflict.split_conflict_geom_data(X_test)

    if not len(X_test_ID) == len(X_test):
        raise AssertionError('lenght X_test_ID does not match lenght X_test - {} vs {}'.format(len(X_test_ID), len(X_test)))

    if not len(X_test_geom) == len(X_test):
        raise AssertionError('lenght X_test_geom does not match lenght X_test - {} vs {}'.format(len(X_test_geom), len(X_test)))

    return X_train, X_test, y_train, y_test, X_train_geom, X_test_geom, X_train_ID, X_test_ID

def fit_predict(X_train, y_train, X_test, clf, config, pickle_dump=True):
    """Fits the classifier based on training-data and makes predictions.
    Additionally, the prediction probability is determined.

    Args:
        X_train (array): training-data of variable values
        y_train ([type]): training-data of conflict data
        X_test ([type]): test-data of variable values
        clf (classifier): the specified model instance.

    Returns:
        arrays: arrays including the predictions made and their probabilities
    """    

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    y_prob = clf.predict_proba(X_test)

    return y_pred, y_prob

def pickle_clf(scaler, clf, config):
    """(Re)fits a classifier with all available data and pickles it.
    Can then be used to make projections in conjuction with projected values.

    Args:
        scaler (scaler): the specified scaling method instance.
        clf (classifier): the specified model instance.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        classifier: classifier fitted with all available data.
    """    

    print('INFO: fitting the classifier with all data from reference period')

    if config.get('pre_calc', 'XY') is '':
        if config.getboolean('general', 'verbose'): print('DEBUG: loading XY data from {}'.format(os.path.abspath(os.path.join(config.get('general', 'output_dir'), 'XY.npy'))))
        XY_fit = np.load(os.path.abspath(os.path.join(config.get('general', 'output_dir'), 'XY.npy')), allow_pickle=True)
    else:
        if config.getboolean('general', 'verbose'): print('DEBUG: loading XY data from {}'.format(os.path.abspath(config.get('pre_calc', 'XY'))))
        XY_fit = np.load(os.path.abspath(os.path.join(config.get('general', 'output_dir'), config.get('pre_calc', 'XY'))), allow_pickle=True)

    X_fit, Y_fit = data.split_XY_data(XY_fit, config)
    X_ID_fit, X_geom_fit, X_data_fit = conflict.split_conflict_geom_data(X_fit)
    X_ft_fit = scaler.fit_transform(X_data_fit)

    clf.fit(X_ft_fit, Y_fit)

    print('INFO: dumping classifier to {}'.format(os.path.abspath(os.path.join(config.get('general', 'output_dir'), 'clf.pkl'))))
    with open(os.path.abspath(os.path.join(config.get('general', 'output_dir'), 'clf.pkl')), 'wb') as f:
        pickle.dump(clf, f)

    return clf