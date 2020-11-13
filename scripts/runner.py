import copro 

import click
import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

@click.group()
def cli():
    pass

@click.command()
@click.argument('cfg', type=click.Path())
@click.option('--projection-settings', '-proj', help='path to cfg-file with settings for a projection run', multiple=True, type=click.Path())
@click.option('--verbose', '-v', help='command line switch to turn on verbose mode', is_flag=True)

def main(cfg, projection_settings=[], verbose=False):   
    """Main command line script to execute the model. 
    All settings are read from cfg-file.
    One cfg-file is required argument to train, test, and evaluate the model.
    Additional cfg-files can be provided as optional arguments, whereby each file corresponds to one projection to be made.

    Args:
        CFG (str): (relative) path to cfg-file
    """ 
  
    #- parsing settings-file and getting path to output folder
    config, out_dir = copro.utils.initiate_setup(cfg)

    if verbose:
        config.set('general', 'verbose', str(verbose))

    click.echo(click.style('\nINFO: reference run started\n', fg='cyan'))

    #- selecting conflicts and getting area-of-interest and aggregation level
    conflict_gdf, extent_gdf, extent_active_polys_gdf, global_df = copro.selection.select(config, out_dir)
    #- plot selected polygons and conflicts
    fig, ax = plt.subplots(1, 1)
    copro.plots.selected_polygons(extent_active_polys_gdf, figsize=(20, 10), ax=ax)
    copro.plots.selected_conflicts(conflict_gdf, ax=ax)
    plt.savefig(os.path.join(out_dir, 'selected_polygons_and_conflicts.png'), dpi=300, bbox_inches='tight')

    #- create X and Y arrays by reading conflict and variable files;
    #- or by loading a pre-computed array (npy-file)
    X, Y = copro.pipeline.create_XY(config, extent_active_polys_gdf, conflict_gdf)

    #- defining scaling and model algorithms
    scaler, clf = copro.pipeline.prepare_ML(config)

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
    for n in range(config.getint('settings', 'n_runs')):
        
        click.echo('INFO: run {} of {}'.format(n+1, config.getint('settings', 'n_runs')))

        #- run machine learning model and return outputs
        X_df, y_df, eval_dict = copro.pipeline.run_reference(X, Y, config, scaler, clf, out_dir)
        
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
    plt.savefig(os.path.join(out_dir, 'ROC_curve_per_run.png'), dpi=300, bbox_inches='tight')

    #- save output dictionary to csv-file
    copro.utils.save_to_csv(out_dict, out_dir, 'out_dict')
    copro.utils.save_to_npy(out_y_df, out_dir, 'out_y_df')
    
    #- print mean values of all evaluation metrics
    for key in out_dict:
        if config.getboolean('general', 'verbose'):
            click.echo('DEBUG: average {0} of run with {1} repetitions is {2:0.3f}'.format(key, config.getint('settings', 'n_runs'), np.mean(out_dict[key])))

    # create accuracy values per polygon and save to output folder
    df_hit, gdf_hit = copro.evaluation.polygon_model_accuracy(out_y_df, global_df, out_dir)

    # apply k-fold 
    gdf_CCP = copro.evaluation.calc_kFold_polygon_analysis(out_y_df, global_df, out_dir, k=10)

    #- plot distribution of all evaluation metrics
    fig, ax = plt.subplots(1, 1)
    copro.plots.metrics_distribution(out_dict, figsize=(20, 10))
    plt.savefig(os.path.join(out_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(1, 1)
    copro.plots.polygon_categorization(gdf_hit, ax=ax)
    plt.savefig(os.path.join(out_dir, 'polygon_categorization.png'), dpi=300, bbox_inches='tight')

    clf = copro.machine_learning.pickle_clf(scaler, clf, config)
    #- plot relative importance of each feature based on ALL data points
    fig, ax = plt.subplots(1, 1)
    copro.plots.factor_importance(clf, config, ax=ax, figsize=(20, 10))
    plt.savefig(os.path.join(out_dir, 'factor_importance.png'), dpi=300, bbox_inches='tight')

    click.echo('INFO: reference run succesfully finished')

    if projection_settings is not []:

        for proj in projection_settings:

            click.echo(click.style('\nINFO: projection run started, based on {}'.format(os.path.abspath(proj)), fg='cyan'))

            config, out_dir = copro.utils.initiate_setup(proj)

            X = copro.pipeline.create_X(config, extent_active_polys_gdf)

            y_df = copro.pipeline.run_prediction(X, scaler, config)

            df_hit, gdf_hit = copro.evaluation.polygon_model_accuracy(y_df, global_df, out_dir=out_dir, make_proj=True)

if __name__ == '__main__':
    main()