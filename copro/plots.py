import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
sns.set_palette('colorblind')
import numpy as np
import os
from sklearn import metrics
from copro import evaluation
import numpy as np
from sklearn.tree import plot_tree


def selected_polygons(polygon_gdf, **kwargs):
    """Creates a plotting instance of the boundaries of all selected polygons.

    Args:
        polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.

    Kwargs:
        Geopandas-supported keyword arguments.

    Returns:
        ax: Matplotlib axis object.   
    """    

    ax = polygon_gdf.boundary.plot(**kwargs)

    return ax

def selected_migration(migration_gdf, **kwargs):
    """Creates a plotting instance of the best casualties estimates of the selected migration.

    Args:
        migration_gdf (geo-dataframe): geo-dataframe containing the selected net migration.

    Kwargs:
        Geopandas-supported keyword arguments.

    Returns:
        ax: Matplotlib axis object.   
    """       

    ax = migration_gdf.plot(column='best', **kwargs)

    return ax

def metrics_distribution(out_dict, metrics, **kwargs):
    """Plots the value distribution of a range of evaluation metrics based on all model simulations.

    Args:
        out_dict (dict): dictionary containing metrics score for various metrics and all simulation.

    Kwargs:
        Matplotlib-supported keyword arguments.

    Returns:
        ax: Matplotlib axis object.    
    """    

    fig, ax = plt.subplots(1, 1, **kwargs)

    for metric, color in zip(metrics, sns.color_palette('colorblind')):

        sns.histplot(out_dict[str(metric)], ax=ax, kde=True, stat='density', color=color, label=str(metric))

    plt.legend()

    return ax

def correlation_matrix(df, **kwargs):
    """Plots the correlation matrix of a dataframe.

    Args:
        df (dataframe): dataframe containing columns to be correlated.

    Kwargs:
        Seaborn-supported keyword arguments.

    Returns:
        ax: Matplotlib axis object.    
    """    

    df_corr = evaluation.calc_correlation_matrix(df)

    ax = sns.heatmap(df_corr, **kwargs)
    
    return ax

def plot_ROC_curve_n_times(ax, clf, X_test, y_test, tprs, aucs, mean_fpr, **kwargs):
    """Plots the ROC-curve per model simulation to a pre-initiated matplotlib-instance.

    Args:
        ax (axis): axis of pre-initaited matplotlib-instance
        clf (classifier): sklearn-classifier used in the simulation.
        X_test (array): array containing test-sample variable values.
        y_test (list): list containing test-sample migration data.
        tprs (list): list with false positive rates.
        aucs (list): list with area-under-curve values.
        mean_fpr (array): array with mean false positive rate.

    Returns:
        list: lists with true positive rates and area-under-curve values per plot.
    """    

    raise DeprecationWarning('Plotting API in sklearn is changed, function needs updating.')
    
    viz = metrics.plot_roc_curve(clf, X_test, y_test, ax=ax,
                            	 alpha=0.15, color='b', lw=1, label=None, **kwargs)
    
    # rfc_disp = metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax,
    #                                                   alpha=0.15, color='b', lw=1, label=None, **kwargs)

    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

    return tprs, aucs

def plot_ROC_curve_n_mean(ax, tprs, aucs, mean_fpr, **kwargs):
    """Plots the mean ROC-curve to a pre-initiated matplotlib-instance.

    Args:
        ax (axis): axis of pre-initaited matplotlib-instance
        tprs (list): list with false positive rates.
        aucs (list): list with area-under-curve values.
        mean_fpr (array): array with mean false positive rate.

            """    
 