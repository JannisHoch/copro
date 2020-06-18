import conflict_model 

from configparser import RawConfigParser

import click
from os.path import isdir, dirname, abspath
from os import makedirs
import geopandas as gpd
import pandas as pd
import numpy as np
import seaborn as sbs
from sklearn import svm, preprocessing, model_selection, metrics
import matplotlib.pyplot as plt
import os, sys


@click.group()
def cli():
    pass

@click.command()
@click.argument('cfg',)
@click.option('-so', '--safe-output', default=False, help='whether or not to save output', type=click.BOOL)

def main(cfg, safe_output=False):
    """
    Runs the conflict_model from command line with several options and the settings cfg-file as argument.

    CFG: path to cfg-file with run settings
    """
    print('')
    print('#### LETS GET STARTED PEOPLZ! ####' + os.linesep)

    if gpd.__version__ < '0.7.0':
        sys.exit('please upgrade geopandas to version 0.7.0, your current version is {}'.format(gpd.__version__))

    config = RawConfigParser(allow_no_value=True)
    config.read(cfg)

    if safe_output:
        out_dir = config.get('general','output_dir')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        print('saving output to folder {}'.format(out_dir) + os.linesep)
    else:
        print('not saving output' + os.linesep)

    gdf = conflict_model.utils.get_geodataframe(config)

    conflict_gdf, extent_gdf = conflict_model.selection.select(gdf, config)

    print('data retrieval period from', str(config.getint('settings', 'y_start')), 'to', str(config.getint('settings', 'y_end')))
    print('')

    X1 = pd.Series(dtype=float)
    X2 = pd.Series(dtype=float)
    Y  = pd.Series(dtype=int)

    for sim_year in np.arange(config.getint('settings', 'y_start'), config.getint('settings', 'y_end'), 1):

        print('entering year {}'.format(sim_year) + os.linesep)

        list_boolConflict = conflict_model.get_boolean_conflict.conflict_in_year_bool(conflict_gdf, extent_gdf, config, sim_year)
        Y = Y.append(pd.Series(list_boolConflict, dtype=int), ignore_index=True)
        
        list_GDP_PPP = conflict_model.get_var_from_nc.nc_with_integer_timestamp(extent_gdf, config, 'GDP_per_capita_PPP', sim_year)
        X1 = X1.append(pd.Series(list_GDP_PPP), ignore_index=True)

        if not len(list_GDP_PPP) == len(list_boolConflict):
            raise AssertionError('length of lists do not match, they are {0} and {1}'.format(len(list_GDP_PPP), len(list_boolConflict)))

        list_Evap = conflict_model.get_var_from_nc.nc_with_continous_regular_timestamp(extent_gdf, config, 'total_evaporation', sim_year)
        X2 = X2.append(pd.Series(list_Evap), ignore_index=True)

        if not len(list_Evap) == len(list_boolConflict):
            raise AssertionError('length of lists do not match, they are {0} and {1}'.format(len(list_Evap), len(list_boolConflict)))

    print('...all data retrieved' + os.linesep)

    print('preparing data for Machine Learning model' + os.linesep)
    XY_data = list(zip(X1, X2, Y))
    XY_data = pd.DataFrame(XY_data, columns=['GDP_PPP', 'ET', 'conflict'])
    XY_data = XY_data.dropna()
    X = XY_data[['GDP_PPP', 'ET']].to_numpy()
    Y = XY_data.conflict.astype(int).to_numpy()

    print('scaling data and splitting into trainings and test samples' + os.linesep)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(preprocessing.scale(X),
                                                                        Y,
                                                                        test_size=0.7)

    plt.figure(figsize=(10,10))
    sbs.scatterplot(x=X_train[:,0],
                    y=X_train[:,1],  
                    hue=y_train)

    plt.title('n=' + str(len(X_train)))
    if safe_output:
        plt.savefig(os.path.join(out_dir, 'scatter_plot.png'), dpi=300)

    print('initializing Support Vector Classification model' + os.linesep)
    clf = svm.SVC(class_weight='balanced')

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

    disp = metrics.plot_precision_recall_curve(clf, X_test, y_test)
    disp.ax_.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(metrics.average_precision_score(y_test, y_score)))

if __name__ == '__main__':
    main()