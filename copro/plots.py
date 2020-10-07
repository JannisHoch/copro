import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sbs
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

def metrics_distribution(out_dict, **kwargs):
    """Plots the value distribution of a range of evaluation metrics based on all model simulations.

    Args:
        out_dict (dict): dictionary containing metrics score for various metrics and all simulation.

    Kwargs:
        Matplotlib-supported keyword arguments.

    Returns:
        ax: Matplotlib axis object.    
    """    

    fig, ax = plt.subplots(1, 1, **kwargs)

    sbs.displot(out_dict['Accuracy'], ax=ax, color="k", label='Accuracy')
    sbs.displot(out_dict['Precision'], ax=ax, color="r", label='Precision')
    sbs.displot(out_dict['Recall'], ax=ax, color="b", label='Recall')
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

    ax = sbs.heatmap(df_corr, **kwargs)
    
    return ax

def polygon_categorization(gdf, category='sub', method='median', **kwargs):
    """Plots the categorization of polygons based on chance of correct prediction and number of conflicts.

    Main categories are:
        * H: chance of correct prediction higher than treshold;
        * L: chance of correct prediction lower than treshold.

    Sub-categories are:
        * HH: high chance of correct prediction with high number of conflicts;
        * HL: high chance of correct prediction with low number of conflicts;
        * LH: low chance of correct prediction with high number of conflicts;
        * LL: low chance of correct prediction with low number of conflicts.

    Args:
        gdf (geo-dataframe): containing model evaluation per unique polygon.
        out_dir (str): path to output folder
        method (str, optional): Statistical method used to determine categorization threshold. Defaults to 'median'.

    Kwargs:
        Matplotlib-supported keyword arguments.

    Returns:
        ax: Matplotlib axis object.        
    """    

    gdf = evaluation.categorize_polys(gdf, category, method)

    ax = gdf.plot(column='category', **kwargs)

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

def factor_importance(clf, config, out_dir=None, **kwargs):
    """Plots the relative importance of each factor as bar plot. Note, this works only for RFClassifier as ML-model!

    Args:
        clf (classifier): sklearn-classifier used in the simulation.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        out_dir (str): path to output folder. If None, output is not saved.

    Kwargs:
        Matplotlib-supported keyword arguments.

    Returns:
        ax: Matplotlib axis object.
    """    

    df = evaluation.get_feature_importance(clf, config, out_dir)

    ax = df.plot.bar(**kwargs)

    return ax
    
