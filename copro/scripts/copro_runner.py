import copro 

import click
import pandas as pd
import numpy as np
import os, sys
import pickle
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

@click.command()
@click.argument('cfg', type=click.Path())
@click.option('--make_plots', '-plt', help='add additional output plots', type=bool)
@click.option('--verbose', '-v', help='command line switch to turn on verbose mode', is_flag=True)

def cli(cfg, make_plots=True, verbose=False):   
    """Main command line script to execute the model. 
    All settings are read from cfg-file.
    One cfg-file is required argument to train, test, and evaluate the model.
    Multiple classifiers are trained based on different train-test data combinations.
    Additional cfg-files for multiple projections can be provided as optional arguments, whereby each file corresponds to one projection to be made.
    Per projection, each classifiers is used to create separate projection outcomes per time step (year).
    All outcomes are combined after each time step to obtain the common projection outcome.

    Args:
        CFG (str): (relative) path to cfg-file
    """ 
  
    #- parsing settings-file
    #- returns dictionary with config-objects and output directories of reference run and all projections
    #- also returns root_dir which is the path to the cfg-file
    main_dict, root_dir = copro.utils.initiate_setup(cfg)

    #- get config-objct and out_dir for reference run
    config_REF = main_dict['_REF'][0]
    out_dir_REF = main_dict['_REF'][1]

    #- is specified, set verbose-settings
    if verbose:
        config_REF.set('general', 'verbose', str(verbose))

    click.echo(click.style('\nINFO: reference run started\n', fg='cyan'))

    #- selecting conflicts and getting area-of-interest and aggregation level
    conflict_gdf, extent_active_polys_gdf, global_df = copro.selection.select(config_REF, out_dir_REF, root_dir)

    #- plot selected polygons and conflicts
    if make_plots:
        fig, ax = plt.subplots(1, 1)
        copro.plots.selected_polygons(extent_active_polys_gdf, figsize=(20, 10), ax=ax)
        copro.plots.selected_conflicts(conflict_gdf, ax=ax)
        plt.savefig(os.path.join(out_dir_REF, 'selected_polygons_and_conflicts.png'), dpi=300, bbox_inches='tight')

    #- create X and Y arrays by reading conflict and variable files for reference run
    #- or by loading a pre-computed array (npy-file) if specified in cfg-file
    X, Y = copro.pipeline.create_XY(config_REF, out_dir_REF, root_dir, extent_active_polys_gdf, conflict_gdf)

    #- defining scaling and model algorithms
    scaler, clf = copro.pipeline.prepare_ML(config_REF)

    #- fit-transform on scaler to be used later during projections
    click.echo('INFO: fitting scaler to sample data')
    scaler_fitted = scaler.fit(X[: , 2:])

    #- initializing output variables
    #TODO: put all this into one function
    out_X_df = copro.evaluation.init_out_df()
    out_y_df = copro.evaluation.init_out_df()
    out_dict = copro.evaluation.init_out_dict()
    trps, aucs, mean_fpr = copro.evaluation.init_out_ROC_curve()

    #- create plot instance for ROC plots
    fig, ax1 = plt.subplots(1, 1, figsize=(20,10))

    #- go through all n model executions
    #- that is, create different classifiers based on different train-test data combinations
    click.echo('INFO: training and testing machine learning model')
    for n in range(config_REF.getint('machine_learning', 'n_runs')):
        
        click.echo('INFO: run {} of {}'.format(n+1, config_REF.getint('machine_learning', 'n_runs')))

        #- run machine learning model and return outputs
        X_df, y_df, eval_dict = copro.pipeline.run_reference(X, Y, config_REF, scaler, clf, out_dir_REF, run_nr=n+1)
        
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
    if config_REF.getboolean('general', 'verbose'):
        for key in out_dict:
            click.echo('DEBUG: average {0} of run with {1} repetitions is {2:0.3f}'.format(key, config_REF.getint('machine_learning', 'n_runs'), np.mean(out_dict[key])))

    #- create accuracy values per polygon and save to output folder
    #- note only the dataframe is stored, not the geo-dataframe
    df_hit, gdf_hit = copro.evaluation.polygon_model_accuracy(out_y_df, global_df)

    gdf_hit.to_file(os.path.join(out_dir_REF, 'output_for_REF.geojson'), driver='GeoJSON')

    #- plot distribution of all evaluation metrics
    if make_plots:
        fig, ax = plt.subplots(1, 1)
        copro.plots.metrics_distribution(out_dict, figsize=(20, 10))
        plt.savefig(os.path.join(out_dir_REF, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')

    df_feat_imp = copro.evaluation.get_feature_importance(clf, config_REF, out_dir_REF) 
    df_perm_imp = copro.evaluation.get_permutation_importance(clf, scaler.fit_transform(X[:,2:]), Y, df_feat_imp, out_dir_REF)

    click.echo(click.style('\nINFO: reference run succesfully finished\n', fg='cyan'))

    click.echo(click.style('INFO: starting projections\n', fg='cyan'))

    #- running prediction runs
    copro.pipeline.run_prediction(scaler_fitted, main_dict, root_dir, extent_active_polys_gdf)

    click.echo(click.style('\nINFO: all projections succesfully finished\n', fg='cyan'))