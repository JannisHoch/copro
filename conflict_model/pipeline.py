#TODO: if that remains the only function in this py-file, then better move the function in the click-script and execute there

from conflict_model import models, data, machine_learning, evaluation
import numpy as np
import os, sys


def create_XY(config, conflict_gdf, extent_active_polys_gdf):

    if config.get('pre_calc', 'XY') is '':

        XY = data.initiate_XY_data(config)

        XY = data.fill_XY(XY, config, conflict_gdf, extent_active_polys_gdf)

        if config.getboolean('general', 'verbose'): 
            print('loading XY data from file {}'.format(os.path.abspath(os.path.join(config.get('general', 'input_dir'), 'XY.npy'))) + os.linesep)
            np.save(os.path.join(config.get('general', 'input_dir'), 'XY.npy'), XY, allow_pickle=True)

    else:

        if config.getboolean('general', 'verbose'): 
            print('loading XY data from file {}'.format(os.path.abspath(os.path.join(config.get('general', 'input_dir'), config.get('pre_calc', 'XY')))) + os.linesep)

        XY = np.load(os.path.join(config.get('general', 'input_dir'), config.get('pre_calc', 'XY')), allow_pickle=True)

    X, Y = data.split_XY_data(XY, config)

    return X, Y

def prepare_ML(config):

    scaler = machine_learning.define_scaling(config)

    clf = machine_learning.define_model(config)

    return scaler, clf

def run(X, Y, config, scaler, clf, out_dir):

    if config.getint('general', 'model') == 1:
        X_df, y_df, eval_dict = models.all_data(X, Y, config, scaler, clf, out_dir)
    elif config.getint('general', 'model') == 2:
        y_df, y_gdf, eval_dict = models.leave_one_out(X, Y, config, scaler, clf, out_dir)
    elif config.getint('general', 'model') == 3:
        y_df, y_gdf, eval_dict = models.single_variables(X, Y, config, scaler, clf, out_dir)
    elif config.getint('general', 'model') == 4:
        y_df, y_gdf, eval_dict = models.dubbelsteen(X, Y, config, scaler, clf, out_dir)
    else:
        raise ValueError('the specified model type in the cfg-file is invalid - specify either 1, 2, 3 or 4.')

    return X_df, y_df, eval_dict

def evaluate():

    return