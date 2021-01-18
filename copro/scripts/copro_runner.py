import copro 

import click
import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

@click.command()
@click.argument('cfg', type=click.Path())
@click.option('--make_plots', '-plt', help='add additional output plots')
@click.option('--verbose', '-v', help='command line switch to turn on verbose mode', is_flag=True)

def cli(cfg, make_plots=True, verbose=False):   
    """Main command line script to execute the model. 
    All settings are read from cfg-file.
    One cfg-file is required argument to train, test, and evaluate the model.
    Additional cfg-files can be provided as optional arguments, whereby each file corresponds to one projection to be made.

    Args:
        CFG (str): (relative) path to cfg-file
    """ 
  
    #- parsing settings-file and getting path to output folder
    main_dict, root_dir = copro.utils.initiate_setup(cfg)

    config_REF = main_dict['_REF'][0]
    out_dir_REF = main_dict['_REF'][1]

    if verbose:
        config_REF.set('general', 'verbose', str(verbose))

    click.echo(click.style('\nINFO: reference run started\n', fg='cyan'))

    #- selecting conflicts and getting area-of-interest and aggregation level
    conflict_gdf, extent_gdf, extent_active_polys_gdf, global_df = copro.selection.select(config_REF, out_dir_REF, root_dir)

    if make_plots:
        #- plot selected polygons and conflicts
        fig, ax = plt.subplots(1, 1)
        copro.plots.selected_polygons(extent_active_polys_gdf, figsize=(20, 10), ax=ax)
        copro.plots.selected_conflicts(conflict_gdf, ax=ax)
        plt.savefig(os.path.join(out_dir_REF, 'selected_polygons_and_conflicts.png'), dpi=300, bbox_inches='tight')

    #- create X and Y arrays by reading conflict and variable files;
    #- or by loading a pre-computed array (npy-file)
    X, Y = copro.pipeline.create_XY(config_REF, out_dir_REF, root_dir, extent_active_polys_gdf, conflict_gdf)

    #- defining scaling and model algorithms
    scaler, clf = copro.pipeline.prepare_ML(config_REF)

    #- initializing output variables
    #TODO: put all this into one function
    out_X_df = copro.evaluation.init_out_df()
    out_y_df = copro.evaluation.init_out_df()
    out_dict = copro.evaluation.init_out_dict()
    trps, aucs, mean_fpr = copro.evaluation.init_out_ROC_curve()

    #- create plot instance for ROC plots
    fig, ax1 = plt.subplots(1, 1, figsize=(20,10))

    click.echo('INFO: training and testing machine learning model')
    #- go through all n model executions
    for n in range(config_REF.getint('machine_learning', 'n_runs')):
        
        click.echo('INFO: run {} of {}'.format(n+1, config_REF.getint('machine_learning', 'n_runs')))

        #- run machine learning model and return outputs
        X_df, y_df, eval_dict = copro.pipeline.run_reference(X, Y, config_REF, scaler, clf, out_dir_REF)
        
        #- append per model execution
        #TODO: put all this into one function
        out_X_df = copro.evaluation.fill_out_df(out_X_df, X_df)
        out_y_df = copro.evaluation.fill_out_df(out_y_df, y_df)
        out_dict = copro.evaluation.fill_out_dict(out_dict, eval_dict)

        #- plot ROC curve per model execution
        tprs, aucs = copro.plots.plot_ROC_curve_n_times(ax1, clf, X_df.to_numpy(), y_df.y_test.to_list(),
                                                                    trps, aucs, mean_fpr)

    #- plot mean ROC curve
    copro.plots.plot_ROC_curve_n_mean(ax1, tprs, aucs, mean_fpr)
    #- save plot
    plt.savefig(os.path.join(out_dir_REF, 'ROC_curve_per_run.png'), dpi=300, bbox_inches='tight')
    #- save data for plot
    copro.evaluation.save_out_ROC_curve(tprs, aucs, out_dir_REF)

    #- save output dictionary to csv-file
    copro.utils.save_to_csv(out_dict, out_dir_REF, 'evaluation_metrics')
    copro.utils.save_to_npy(out_y_df, out_dir_REF, 'raw_output_data')
    
    #- print mean values of all evaluation metrics
    for key in out_dict:
        if config_REF.getboolean('general', 'verbose'):
            click.echo('DEBUG: average {0} of run with {1} repetitions is {2:0.3f}'.format(key, config_REF.getint('machine_learning', 'n_runs'), np.mean(out_dict[key])))

    # create accuracy values per polygon and save to output folder
    df_hit, gdf_hit = copro.evaluation.polygon_model_accuracy(out_y_df, global_df, out_dir_REF)

    #- plot distribution of all evaluation metrics
    if make_plots:
        fig, ax = plt.subplots(1, 1)
        copro.plots.metrics_distribution(out_dict, figsize=(20, 10))
        plt.savefig(os.path.join(out_dir_REF, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')

    clf = copro.machine_learning.pickle_clf(scaler, clf, config_REF, root_dir)
    #- plot relative importance of each feature based on ALL data points
    if make_plots:
        fig, ax = plt.subplots(1, 1)
        copro.plots.factor_importance(clf, config_REF, out_dir=out_dir_REF, ax=ax, figsize=(20, 10))
        plt.savefig(os.path.join(out_dir_REF, 'feature_importances.png'), dpi=300, bbox_inches='tight')

    click.echo('INFO: reference run succesfully finished')

    sys.exit()

    if projection_settings is not []:

        for proj in projection_settings:

            click.echo(click.style('\nINFO: projection run started, based on {}'.format(os.path.abspath(proj)), fg='cyan'))

            config, out_dir, root_dir = copro.utils.initiate_setup(proj)

            X = copro.pipeline.create_X(config, out_dir, root_dir, extent_active_polys_gdf)

            y_df = copro.pipeline.run_prediction(X, scaler, config, root_dir)

            df_hit, gdf_hit = copro.evaluation.polygon_model_accuracy(y_df, global_df, out_dir=out_dir, make_proj=True)