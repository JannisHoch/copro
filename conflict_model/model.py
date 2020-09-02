#TODO: if that remains the only function in this py-file, then better move the function in the click-script and execute there

from conflict_model import pipelines, data, machine_learning
import os, sys


def prepare_XY(config, conflict_gdf, extent_active_polys_gdf):

    XY = data.initiate_XY_data(config)

    XY = data.fill_XY(XY, config, conflict_gdf, extent_active_polys_gdf)

    X, Y = data.split_XY_data(XY, config)

    return X, Y

def prepare_ML(config):

    scaler = machine_learning.define_scaling(config)

    clf = machine_learning.define_model(config)

    return scaler, clf

def run(X, Y, config, scaler, clf, out_dir):

    if config.getint('general', 'model') == 1:
        y_df, y_gdf, eval_dict = pipelines.all_data(X, Y, config, scaler, clf, out_dir)
    elif config.getint('general', 'model') == 2:
        y_df, y_gdf, eval_dict = pipelines.leave_one_out(X, Y, config, scaler, clf, out_dir)
    elif config.getint('general', 'model') == 3:
        y_df, y_gdf, eval_dict = pipelines.single_variables(X, Y, config, scaler, clf, out_dir)
    elif config.getint('general', 'model') == 4:
        y_df, y_gdf, eval_dict = pipelines.dubbelsteen(X, Y, config, scaler, clf, out_dir)
    else:
        raise ValueError('the specified model type in the cfg-file is invalid - specify either 1, 2, 3 or 4.')

    return y_df, y_gdf, eval_dict

def evaluate():

    return