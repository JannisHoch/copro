import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sbs
import numpy as np
import os, sys
from sklearn import metrics
from conflict_model import evaluation

def plot_active_polys(conflict_gdf, polygon_gdf, config, out_dir, **kwargs):
    """Creates a (1,2)-subplot showing a) selected conflicts and all polygons, and b) selected conflicts and selected polygons.

    Args:
        conflict_gdf (geo-dataframe): geo-dataframe containing the selected conflicts.
        extent_gdf (geo-dataframe): geo-dataframe containing all polygons.
        polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        out_dir (str): path to output folder
    """    

    fig, ax = plt.subplots(1, 1, **kwargs)
    fig.suptitle('conflict distribution; # conflicts {}; threshold casualties {}; type of violence {}'.format(len(conflict_gdf), config.get('conflict', 'min_nr_casualties'), config.get('conflict', 'type_of_violence')))

    conflict_gdf.plot(ax=ax, c='r', column='best', cmap='magma', 
                      vmin=int(config.get('conflict', 'min_nr_casualties')), vmax=conflict_gdf.best.mean(), 
                      legend=True, 
                      legend_kwds={'label': "# casualties",})
    polygon_gdf.boundary.plot(ax=ax)
                            
    plt.savefig(os.path.join(out_dir, 'selected_conflicts_and_polygons.png'), dpi=300)

    return

def plot_metrics_distribution(out_dict, out_dir, **kwargs):
    """Plots the value distribution of a range of evaluation metrics based on all model simulations.

    Args:
        out_dict (dict): dictionary containing metrics score for various metrics and all simulation.
        out_dir (str): path to output folder.
    """    

    fig, ax = plt.subplots(1, 1, **kwargs)

    sbs.distplot(out_dict['Accuracy'], ax=ax, color="k", label='Accuracy')
    sbs.distplot(out_dict['Precision'], ax=ax, color="r", label='Precision')
    sbs.distplot(out_dict['Recall'], ax=ax, color="b", label='Recall')

    plt.savefig(os.path.join(out_dir, 'metrics_distribution.png'), dpi=300)

    return

def plot_nr_and_dist_pred(df, gdf, polygon_gdf, out_dir, suffix='', **kwargs):
    """Plots the number of number of predictions made per unique polygon, and the overall value distribution.

    Args:
        df (dataframe): containing model evaluation per unique polygon.
        gdf (geo-dataframe): containing model evaluation per unique polygon.
        polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.
        out_dir (str): path to output folder.
        suffix (str, optional): suffix that can be used to discriminate between different analyses. Defaults to ''.
    """    

    fig, (ax1, ax2) = plt.subplots(1, 2, **kwargs)

    gdf.plot(ax=ax1, column='ID_count', legend=True, cmap='cool')
    polygon_gdf.boundary.plot(ax=ax1, color='0.5')
    ax1.set_title('number of predictions made per polygon')
    sbs.distplot(df.ID_count.values, ax=ax2)
    ax2.set_title('distribution of predictions')

    plt.savefig(os.path.join(out_dir, 'analyis_predictions' + str(suffix) + '.png'), dpi=300)

    return

def plot_predictiveness(gdf, polygon_gdf, out_dir, suffix=''):
    """Creates (1,3)-subplot showing per polygon the chance of correct prediction, the number of conflicts, and the chance of correct conflict prediction.

    Args:
        gdf (geo-dataframe): containing model evaluation per unique polygon.
        polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.
        out_dir (str): path to output folder.
        suffix (str, optional): suffix that can be used to discriminate between different analyses. Defaults to ''.
    """   

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    gdf.plot(ax=ax1, column='chance_correct_pred', legend=True, 
                 legend_kwds={'label': "chance correct prediction", 'orientation': "horizontal"})
    polygon_gdf.boundary.plot(ax=ax1, color='0.5')
    gdf.plot(ax=ax2, column='nr_test_confl', legend=True, cmap='Reds', 
                 legend_kwds={'label': "nr of conflicts per polygon", 'orientation': "horizontal"})
    polygon_gdf.boundary.plot(ax=ax2, color='0.5')
    gdf.plot(ax=ax3, column='chance_correct_confl_pred', legend=True, cmap='Blues', 
                 legend_kwds={'label': "chance correct conflict prediction", 'orientation': "horizontal"})
    polygon_gdf.boundary.plot(ax=ax3, color='0.5')

    plt.savefig(os.path.join(out_dir, 'model_predictivness_{}.png'.format(suffix)), dpi=300)

    return

def plot_correlation_matrix(df, out_dir):
    """Plots the correlation matrix of a dataframe.

    Args:
        df (dataframe): dataframe containing columns to be correlated.
        out_dir (str): path to output folder
    """    

    df_corr = evaluation.correlation_matrix(df)

    fig, (ax1) = plt.subplots(1, 1, figsize=(20, 10))
    sbs.heatmap(df_corr, cmap='YlGnBu', annot=True, cbar=False, ax=ax1)
    plt.savefig(os.path.join(out_dir, 'correlation_matrix.png'), dpi=300)

def plot_categories(gdf, out_dir, category='sub', mode='median'):
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
        mode (str, optional): Statistical mode used to determine categorization threshold. Defaults to 'median'.
    """    

    gdf = evaluation.categorize_polys(gdf, category, mode)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    gdf.plot(column='category', categorical=True, legend=True, ax=ax, cmap='copper')

    plt.savefig(os.path.join(out_dir, 'polygon_categorization.png'), dpi=300)

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

    return

def plot_kFold_polygon_analysis(gdf, out_dir, **kwargs):
    """Determines the mean and standard deviation of correct chance of prediction (CCP) per polygon.

    Args:
        gdf (geodataframe): geodataframe containing statistical information.
        out_dir (str): path to output folder.
    """    

    fig, (ax1, ax2) = plt.subplots(1, 2, **kwargs)
    gdf.plot(column='mean_CCP', ax=ax1, legend=True)
    ax1.set_title('MEAN')
    gdf.plot(column='std_CCP', ax=ax2, legend=True)
    ax2.set_title('STD')
    
    plt.savefig(os.path.join(out_dir, 'mean_and_std_CCP.png'), dpi=300)

    return
    
