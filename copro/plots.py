import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
sns.set_palette('colorblind')
import numpy as np
import os, sys
from sklearn import metrics
from copro import evaluation

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

def selected_conflicts(conflict_gdf, **kwargs):
    """Creates a plotting instance of the best casualties estimates of the selected conflicts.

    Args:
        conflict_gdf (geo-dataframe): geo-dataframe containing the selected conflicts.

    Kwargs:
        Geopandas-supported keyword arguments.

    Returns:
        ax: Matplotlib axis object.   
    """       

    ax = conflict_gdf.plot(column='best', **kwargs)

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
        y_test (list): list containing test-sample conflict data.
        tprs (list): list with false positive rates.
        aucs (list): list with area-under-curve values.
        mean_fpr (array): array with mean false positive rate.

    Returns:
        list: lists with true positive rates and area-under-curve values per plot.
    """    

    viz = metrics.plot_roc_curve(clf, X_test, y_test, ax=ax,
                            	 alpha=0.15, color='b', lw=1, label=None, **kwargs)

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

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='r',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8, **kwargs)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=None, **kwargs)

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], **kwargs)

    ax.legend(loc="lower right")
    
