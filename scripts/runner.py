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
    print('#### LETS GET STARTED PEOPLZ! ####' + os.linesep)

    #- parsing settings-file and getting path to output folder
    config, out_dir = conflict_model.utils.initiate_setup(cfg)

    print('verbose mode on: {}'.format(config.getboolean('general', 'verbose')) + os.linesep)

    #- selecting conflicts and getting area-of-interest and aggregation level
    conflict_gdf, extent_gdf, extent_active_polys_gdf, global_df = conflict_model.selection.select(config)
    #- plot selected conflicts and polygons
    conflict_model.plots.plot_active_polys(conflict_gdf, extent_gdf, extent_active_polys_gdf, config, out_dir)

    #- create X and Y arrays by reading conflict and variable files;
    #- or by loading a pre-computed array (npy-file)
    X, Y = conflict_model.pipeline.create_XY(config, conflict_gdf, extent_active_polys_gdf)

    #- defining scaling and model algorithms
    scaler, clf = conflict_model.pipeline.prepare_ML(config)

    #- initializing output variables
    #TODO: put all this into one function
    out_X_df = conflict_model.evaluation.init_out_df()
    out_X1_df = conflict_model.evaluation.init_out_df()
    out_y_df = conflict_model.evaluation.init_out_df()
    out_y1_df = conflict_model.evaluation.init_out_df()
    out_dict = conflict_model.evaluation.init_out_dict()
    trps, aucs, mean_fpr = conflict_model.evaluation.init_out_ROC_curve()

    #- create plot instance for ROC plots
    fig, (ax1) = plt.subplots(1, 1, figsize=(20,10))

    #- go through all n model executions
    for n in range(config.getint('settings', 'n_runs')):
        
        if config.getboolean('general', 'verbose'):
            print('run {} of {}'.format(n+1, config.getint('settings', 'n_runs')) + os.linesep)

        #- run machine learning model and return outputs
        X_df, y_df, eval_dict = conflict_model.pipeline.run(X, Y, config, scaler, clf, out_dir)
        
        #- select sub-dataset with only datapoints with observed conflicts
        X1_df, y1_df = conflict_model.utils.get_conflict_datapoints_only(X_df, y_df)
        
        #- append per model execution
        #TODO: put all this into one function
        out_X_df = conflict_model.evaluation.fill_out_df(out_X_df, X_df)
        out_X1_df = conflict_model.evaluation.fill_out_df(out_X1_df, X1_df)
        out_y_df = conflict_model.evaluation.fill_out_df(out_y_df, y_df)
        out_y1_df = conflict_model.evaluation.fill_out_df(out_y1_df, y1_df)
        out_dict = conflict_model.evaluation.fill_out_dict(out_dict, eval_dict)

        #- plot ROC curve per model execution
        tprs, aucs = conflict_model.evaluation.plot_ROC_curve_n_times(ax1, clf, X_df.to_numpy(), y_df.y_test.to_list(),
                                                                    trps, aucs, mean_fpr)

    #- plot mean ROC curve
    conflict_model.evaluation.plot_ROC_curve_n_mean(ax1, tprs, aucs, mean_fpr)
    #- save plot
    plt.savefig(os.path.join(out_dir, 'ROC_curve_per_run.png'), dpi=300)

    #- print mean values of all evaluation metrics
    for key in out_dict:
        if config.getboolean('general', 'verbose'):
            print('average {0} of run with {1} repetitions is {2:0.3f}'.format(key, config.getint('settings', 'n_runs'), np.mean(out_dict[key])))

    #- plot distribution of all evaluation metrics
    conflict_model.plots.plot_metrics_distribution(out_dict, out_dir)

    #- compute average correct prediction per polygon for all data points as well as conflicty-only
    df_hit, gdf_hit = conflict_model.evaluation.get_average_hit(out_y_df, global_df)
    df_hit_1, gdf_hit_1 = conflict_model.evaluation.get_average_hit(out_y1_df, global_df)

    #- for both, plot number of predictions made per polygon and overall distribution
    conflict_model.plots.plot_nr_and_dist_pred(df_hit, gdf_hit, extent_active_polys_gdf, out_dir)
    conflict_model.plots.plot_nr_and_dist_pred(df_hit_1, gdf_hit_1, extent_active_polys_gdf, out_dir, suffix='conflicts_only')

    #- for both, plot average correct prediction and number of conflicht per polygon
    conflict_model.plots.plot_frac_and_nr_conf(gdf_hit, extent_active_polys_gdf, out_dir)
    conflict_model.plots.plot_frac_and_nr_conf(gdf_hit_1, extent_active_polys_gdf, out_dir, suffix='conflicts_only')

    #- for both, plot distribution of average correct predictions
    conflict_model.plots.plot_frac_pred(gdf_hit, gdf_hit_1, out_dir)

    conflict_model.plots.plot_scatterdata(df_hit, out_dir)

    #- save some dataframes to file
    df_hit.to_csv(os.path.join(out_dir, 'df_hit.csv'))
    pd.DataFrame.from_dict(out_dict).to_csv(os.path.join(out_dir, 'out_dict.csv'), index=False)

if __name__ == '__main__':
    main()