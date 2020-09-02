#TODO: if that remains the only function in this py-file, then better move the function in the click-script and execute there

from conflict_model import pipelines
import os, sys

def run(X, Y, config, scalers, clfs, out_dir):

    if config.getint('general', 'model') == 1:
        y_df, y_gdf, eval_dict = pipelines.all_data(X, Y, config, scalers, clfs, out_dir)
    elif config.getint('general', 'model') == 2:
        y_df, y_gdf, eval_dict = pipelines.leave_one_out(X, Y, config, scalers, clfs, out_dir)
    elif config.getint('general', 'model') == 3:
        y_df, y_gdf, eval_dict = pipelines.single_variables(X, Y, config, scalers, clfs, out_dir)
    elif config.getint('general', 'model') == 4:
        y_df, y_gdf, eval_dict = pipelines.dubbelsteen(X, Y, config, scalers, clfs, out_dir)
    else:
        raise ValueError('the specified model type in the cfg-file is invalid - specify either 1, 2, 3 or 4.')

    return y_df, y_gdf, eval_dict