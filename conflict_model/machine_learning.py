import os
import pandas as pd
from sklearn import svm, neighbors, ensemble, preprocessing

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

def initiate_XY_data(config):
    """[summary]

    Args:
        config ([type]): [description]

    Returns:
        [type]: [description]
    """    

    XY = {}
    for key in config.items('env_vars'):
        XY[str(key[0])] = pd.Series(dtype=float)
    XY['conflict'] = pd.Series(dtype=int)
    XY['conflict_geometry'] = pd.Series()

    return XY

def prepare_XY_data(XY):
    """[summary]

    Args:
        XY ([type]): [description]

    Returns:
        [type]: [description]
    """    

    XY = pd.DataFrame.from_dict(XY)
    print('number of data points including missing values:', len(XY))
    XY = XY.dropna()
    print('number of data points excluding missing values:', len(XY))

    X = XY.to_numpy()[:, :-2] # since conflict is the last column, we know that all previous columns must be variable values
    Y = XY.conflict.astype(int).to_numpy()
    Y_geom = XY.conflict_geometry.to_numpy()

    return X, Y, Y_geom