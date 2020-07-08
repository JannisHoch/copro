import conflict_model 

from configparser import RawConfigParser

import click
from os.path import isdir, dirname, abspath
from os import makedirs
import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np
import seaborn as sbs
from sklearn import svm, preprocessing, model_selection, metrics
import matplotlib.pyplot as plt
from shutil import copyfile
import csv
import os, sys


@click.group()
def cli():
    pass

@click.command()
@click.argument('cfg',)
@click.option('-so', '--safe-output', default=True, help='save output yes/no', type=click.BOOL)
@click.option('-o', '--output-folder', default=None, help='output folder', type=click.Path())
@click.option('-v', '--verbose', default=False, help='verbose model yes/no', is_flag=True, type=click.BOOL)

def main(cfg, safe_output=True, output_folder=None, verbose=False):
    """
    Runs the conflict_model from command line with several options and the settings cfg-file as argument.

    CFG: path to cfg-file with run settings
    """
    print('')
    print('#### LETS GET STARTED PEOPLZ! ####' + os.linesep)

    print('safe output: {}'.format(safe_output))
    print('verbose mode on: {}'.format(verbose) + os.linesep)

    if gpd.__version__ < '0.7.0':
        sys.exit('please upgrade geopandas to version 0.7.0, your current version is {}'.format(gpd.__version__))

    config = RawConfigParser(allow_no_value=True)
    config.optionxform = lambda option: option
    config.read(cfg)

    if safe_output:
        if output_folder == None:
            out_dir = os.path.abspath(config.get('general','output_dir'))
        else:
            out_dir = output_folder
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        print('saving output to folder {}'.format(out_dir) + os.linesep)
    else:
        print('not saving output' + os.linesep)

    if verbose: copyfile(cfg, os.path.join(out_dir, 'copy_of_run_setting.cfg'))

    gdf = conflict_model.utils.get_geodataframe(config)

    conflict_gdf, extent_gdf = conflict_model.selection.select(gdf, config)

    print('data retrieval period from', str(config.getint('settings', 'y_start')), 'to', str(config.getint('settings', 'y_end')))
    print('')

    XY = {}
    for key in config.items('env_vars'):
        XY[str(key[0])] = pd.Series(dtype=float)
    XY['conflict'] = pd.Series(dtype=int)    

    for sim_year in np.arange(config.getint('settings', 'y_start'), config.getint('settings', 'y_end'), 1):
    
        print('entering year {}'.format(sim_year) + os.linesep)
        
        # go through all keys in dictionary
        for key, value in XY.items():
            
            if key == 'conflict':
                data_series = value
                data_list = conflict_model.get_boolean_conflict.conflict_in_year_bool(conflict_gdf, extent_gdf, config, sim_year)
                data_series = data_series.append(pd.Series(data_list), ignore_index=True)
                XY[key] = data_series
                
            else:
                nc_fo = os.path.join(config.get('general', 'input_dir'), 
                                    config.get('env_vars', key))
                
                print('calculating mean {0} per aggregation unit from file {1} for year {2}'.format(key, nc_fo, sim_year))

                nc_ds = xr.open_dataset(nc_fo)
                
                if (np.dtype(nc_ds.time) == np.float32) or (np.dtype(nc_ds.time) == np.float64):
                    data_series = value
                    data_list = conflict_model.get_var_from_nc.nc_with_float_timestamp(extent_gdf, config, key, sim_year)
                    data_series = data_series.append(pd.Series(data_list), ignore_index=True)
                    XY[key] = data_series
                    
                elif np.dtype(nc_ds.time) == 'datetime64[ns]':
                    data_series = value
                    data_list = conflict_model.get_var_from_nc.nc_with_continous_datetime_timestamp(extent_gdf, config, key, sim_year)
                    data_series = data_series.append(pd.Series(data_list), ignore_index=True)
                    XY[key] = data_series
                    
                else:
                    raise Warning('this nc-file does have a different dtype for the time variable than currently supported: {}'.format(nc_fo))
                
    print('...all data retrieved' + os.linesep)

    print('preparing data for Machine Learning model' + os.linesep)
    XY = pd.DataFrame.from_dict(XY).dropna()

    X = XY.to_numpy()[:, :-1]
    Y = XY.conflict.astype(int).to_numpy()

    scaler = preprocessing.QuantileTransformer()

    if verbose:
        scaler_params = scaler.get_params()
        out_fo = os.path.join(out_dir, '{}_params.csv'.format(str(scaler).rsplit('(')[0]))
        w = csv.writer(open(out_fo, "w"))
        for key, val in scaler_params.items():
            w.writerow([key, val])

    print('scaling data with {}'.format(str(scaler).rsplit('(')[0]))
    X_scaled = scaler.fit_transform(X)

    print('splitting into trainings and test samples' + os.linesep)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaled,
                                                                        Y,
                                                                        test_size=1-config.getfloat('machine_learning', 'train_fraction'))

    plt.figure(figsize=(10,10))
    sbs.scatterplot(x=X_train[:,0],
                    y=X_train[:,1],  
                    hue=y_train)

    plt.title('training-data scaled with {0}; n_train={1}; n_tot={2}'.format(str(scaler).rsplit('(')[0], len(X_train), len(X_scaled)))
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    if safe_output: plt.savefig(os.path.join(out_dir, 'scatter_plot_scaled_traindata_{}.png'.format(str(scaler).rsplit('(')[0])), dpi=300)

    print('initializing Support Vector Classification model' + os.linesep)
    clf = svm.LinearSVC(class_weight={1:100}, random_state=42, max_iter=10000000)

    if verbose:
        SVC_params = clf.get_params()
        out_fo = os.path.join(out_dir, 'SVC_params.csv')
        w = csv.writer(open(out_fo, "w"))
        for key, val in SVC_params.items():
            w.writerow([key, val])

    print('fitting model with trainings data' + os.linesep)
    clf.fit(X_train, y_train)

    print('making a prediction' + os.linesep)
    y_pred = clf.predict(X_test)

    y_score = clf.decision_function(X_test)

    print('Model evaluation')
    print("...Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("...Precision:", metrics.precision_score(y_test, y_pred))
    print("...Recall:", metrics.recall_score(y_test, y_pred))
    print('...Average precision-recall score: {0:0.2f}'.format(metrics.average_precision_score(y_test, y_score)))

    fig, ax = plt.subplots(1, 1, figsize=(20,10))
    disp = metrics.plot_precision_recall_curve(clf, X_test, y_test, ax=ax)
    disp.ax_.set_title('2-class Precision-Recall curve: AP={} with {}'.format(round(metrics.average_precision_score(y_test, y_score),2), str(scaler).rsplit('(')[0]))
    if safe_output: plt.savefig(os.path.join(out_dir, 'precision_recall_curve_{}.png'.format(str(scaler).rsplit('(')[0])), dpi=300)

    if safe_output:
        evaluation = {'Accuracy': round(metrics.accuracy_score(y_test, y_pred), 2),
                      'Precision': round(metrics.precision_score(y_test, y_pred), 2),
                      'Recall': round(metrics.recall_score(y_test, y_pred), 2),
                      'Average precision-recall score': round(metrics.average_precision_score(y_test, y_score), 2)}

        out_fo = os.path.join(out_dir, 'evaluation.csv')
        w = csv.writer(open(out_fo, "w"))
        for key, val in evaluation.items():
            w.writerow([key, val])

if __name__ == '__main__':
    main()