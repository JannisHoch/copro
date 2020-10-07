import conflict_model 

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

def main(cfg):   
    """Main command line script to execute the model. All settings are read from cfg-file.

    Args:
        CFG (str): (relative) path to cfg-file
    """    

    print('')
    print('#### CONFLICT MODEL version {} ####'.format(conflict_model.__version__))
    print('')

    #- parsing settings-file and getting path to output folder
    config, out_dir = conflict_model.utils.initiate_setup(cfg)

    print('verbose mode on: {}'.format(config.getboolean('general', 'verbose')) + os.linesep)

    #- selecting conflicts and getting area-of-interest and aggregation level
    conflict_gdf, extent_gdf, extent_active_polys_gdf, global_df = conflict_model.selection.select(config, out_dir)
    #- plot selected polygons and conflicts
    fig, ax = plt.subplots(1, 1)
    conflict_model.plots.selected_polygons(extent_active_polys_gdf, figsize=(20, 10), ax=ax)
    conflict_model.plots.selected_conflicts(conflict_gdf, ax=ax)
    plt.savefig(os.path.join(out_dir, 'selected_polygons_and_conflicts.png'), dpi=300, bbox_inches='tight')

    #- create X and Y arrays by reading conflict and variable files;
    #- or by loading a pre-computed array (npy-file)
    X, Y = conflict_model.pipeline.create_XY(config, conflict_gdf, extent_active_polys_gdf)

    #- defining scaling and model algorithms
    scaler, clf = conflict_model.pipeline.prepare_ML(config)

    #- initializing output variables
    #TODO: put all this into one function
    out_X_df = conflict_model.evaluation.init_out_df()
    out_y_df = conflict_model.evaluation.init_out_df()
    out_dict = conflict_model.evaluation.init_out_dict()
    trps, aucs, mean_fpr = conflict_model.evaluation.init_out_ROC_curve()

    #- create plot instance for ROC plots
    fig, ax1 = plt.subplots(1, 1, figsize=(20,10))

    #- go through all n model executions
    for n in range(config.getint('settings', 'n_runs')):
        
        if config.getboolean('general', 'verbose'):
            print('run {} of {}'.format(n+1, config.getint('settings', 'n_runs')) + os.linesep)

        #- run machine learning model and return outputs
        X_df, y_df, eval_dict = conflict_model.pipeline.run(X, Y, config, scaler, clf, out_dir)
        
        #- append per model execution
        #TODO: put all this into one function
        out_X_df = conflict_model.evaluation.fill_out_df(out_X_df, X_df)
        out_y_df = conflict_model.evaluation.fill_out_df(out_y_df, y_df)
        out_dict = conflict_model.evaluation.fill_out_dict(out_dict, eval_dict)

        #- plot ROC curve per model execution
        tprs, aucs = conflict_model.plots.plot_ROC_curve_n_times(ax1, clf, X_df.to_numpy(), y_df.y_test.to_list(),
                                                                    trps, aucs, mean_fpr)

    #- plot mean ROC curve
    conflict_model.plots.plot_ROC_curve_n_mean(ax1, tprs, aucs, mean_fpr)
    #- save plot
    plt.savefig(os.path.join(out_dir, 'ROC_curve_per_run.png'), dpi=300, bbox_inches='tight')

    #- save output dictionary to csv-file
    conflict_model.utils.save_to_csv(out_dict, out_dir, 'out_dict')
    conflict_model.utils.save_to_npy(out_y_df, out_dir, 'out_y_df')
    
    #- print mean values of all evaluation metrics
    for key in out_dict:
        if config.getboolean('general', 'verbose'):
            print('average {0} of run with {1} repetitions is {2:0.3f}'.format(key, config.getint('settings', 'n_runs'), np.mean(out_dict[key])))

    df_hit, gdf_hit = conflict_model.evaluation.polygon_model_accuracy(out_y_df, global_df, out_dir=None)

    #- plot distribution of all evaluation metrics
    fig, ax = plt.subplots(1, 1)
    conflict_model.plots.metrics_distribution(out_dict, figsize=(20, 10))
    plt.savefig(os.path.join(out_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')

    #- plot relative importance of each feature
    fig, ax = plt.subplots(1, 1)
    conflict_model.plots.factor_importance(clf, config, ax=ax, figsize=(20, 10))
    plt.savefig(os.path.join(out_dir, 'factor_importance.png'), dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(1, 1)
    conflict_model.plots.polygon_categorization(gdf_hit, ax=ax)
    plt.savefig(os.path.join(out_dir, 'polygon_categorization.png'), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()