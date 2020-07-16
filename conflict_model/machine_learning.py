import os
from sklearn import svm, neighbors, preprocessing

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
                neighbors.KNeighborsClassifier(n_neighbors=10, weights='distance')]
    
    elif not config.getboolean('general', 'sensitivity_analysis'):
        if config.get('machine_learning', 'model') == 'NuSVC':
            clfs = [svm.NuSVC(nu=0.1, kernel='rbf', class_weight={1: 100}, random_state=42, probability=True, degree=10, gamma=10)]
        elif config.get('machine_learning', 'model') == 'KNeighborsClassifier':
            clfs = [neighbors.KNeighborsClassifier(n_neighbors=10, weights='distance')]
        else:
            raise ValueError('no supported ML model selected - choose between NuSVC or KNeighborsClassifier')

    print('chosen ML model is {}'.format(clfs[0]))

    return clfs