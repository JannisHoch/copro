import os, sys
from sklearn import metrics
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

def init_out_dict():
    """Initiates the main model evaluatoin dictionary for a range of model metric scores. 
    The scores should match the scores used in the dictioary created in 'evaluation.evaluate_prediction()'.

    Returns:
        dict: empty dictionary with metrics as keys.
    """    

    scores = ['Accuracy', 'Precision', 'Recall', 'F1 score', 'Cohen-Kappa score', 'Brier loss score', 'ROC AUC score']

    out_dict = {}
    for score in scores:
        out_dict[score] = list()

    return out_dict

def evaluate_prediction(y_test, y_pred, y_prob, X_test, clf, config):
    """Computes a range of model evaluation metrics and appends the resulting scores to a dictionary.
    This is done for each model execution separately.

    Args:
        y_test (list): list containing test-sample conflict data.
        y_pred (list): list containing predictions.
        y_prob (array): array resulting probabilties of predictions.
        X_test (array): array containing test-sample variable values.
        clf (classifier): sklearn-classifier used in the simulation.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        dict: dictionary with scores for each simulation
    """  

    if config.getboolean('general', 'verbose'):
        print("Accuracy: {0:0.3f}".format(metrics.accuracy_score(y_test, y_pred)))
        print("Precision: {0:0.3f}".format(metrics.precision_score(y_test, y_pred)))
        print("Recall: {0:0.3f}".format(metrics.recall_score(y_test, y_pred)))
        print('F1 score: {0:0.3f}'.format(metrics.f1_score(y_test, y_pred)))
        print('Brier loss score: {0:0.3f}'.format(metrics.brier_score_loss(y_test, y_prob[:, 1])))
        print('Cohen-Kappa score: {0:0.3f}'.format(metrics.cohen_kappa_score(y_test, y_pred)))
        print('ROC AUC score {0:0.3f}'.format(metrics.roc_auc_score(y_test, y_prob[:, 1])))
        print('')

        print(metrics.classification_report(y_test, y_pred))
        print('')

    eval_dict = {'Accuracy': metrics.accuracy_score(y_test, y_pred),
                 'Precision': metrics.precision_score(y_test, y_pred),
                 'Recall': metrics.recall_score(y_test, y_pred),
                 'F1 score': metrics.f1_score(y_test, y_pred),
                 'Cohen-Kappa score': metrics.cohen_kappa_score(y_test, y_pred),
                 'Brier loss score': metrics.brier_score_loss(y_test, y_prob[:, 1]),
                 'ROC AUC score': metrics.roc_auc_score(y_test, y_prob[:, 1]),
                }

    return eval_dict

def fill_out_dict(out_dict, eval_dict):
    """Appends the computed metric score per run to the main output dictionary.

    Args:
        out_dict (dict): main output dictionary.
        eval_dict (dict): dictionary containing scores per simulation.

    Returns:
        dict: dictionary with collected scores for each simulation
    """    

    for key in out_dict:
        out_dict[key].append(eval_dict[key])

    return out_dict

def init_out_df():
    """Initiates and empty main output dataframe.

    Returns:
        dataframe: empty dataframe.
    """    

    return pd.DataFrame()

def fill_out_df(out_df, y_df):
    """Appends output dataframe of each simulation to main output dataframe.

    Args:
        out_df (dataframe): main output dataframe.
        y_df (dataframe): output dataframe of each simulation.

    Returns:
        dataframe: main output dataframe containing results of all simulations.
    """    

    out_df = out_df.append(y_df, ignore_index=True)

    return out_df

def polygon_model_accuracy(df, global_df):
    """Determines a range of model accuracy values for each polygon.
    Reduces dataframe with results from each simulation to values per unique polygon identifier.
    Determines the total number of predictions made per polygon as well as fraction of correct predictions made for overall and conflict-only data.

    Args:
        df (dataframe): output dataframe containing results of all simulations.
        global_df (dataframe): global look-up dataframe to associate unique identifier with geometry.

    Returns:
        (geo-)dataframe: dataframe and geo-dataframe with data per polygon.
    """    

    #- create a dataframe containing the number of occurence per ID
    ID_count = df.ID.value_counts().to_frame().rename(columns={'ID':'ID_count'})
    #- add column containing the IDs
    ID_count['ID'] = ID_count.index.values
    #- set index with index named ID now
    ID_count.set_index(ID_count.ID, inplace=True)
    #- remove column ID
    ID_count = ID_count.drop('ID', axis=1)

    #- per polygon ID, compute sum of overall correct predictions and rename column name
    hit_count = df.correct_pred.groupby(df.ID).sum().to_frame()

    #- per polygon ID, compute sum of all conflict data points and add to dataframe
    hit_count['nr_test_confl'] = df.y_test.groupby(df.ID).sum()

    #- per polygon ID, compute sum of all conflict data points and add to dataframe
    hit_count['nr_pred_confl'] = df.y_pred.groupby(df.ID).sum()

    #- merge the two dataframes with ID as key
    df_temp = pd.merge(ID_count, hit_count, on='ID')

    #- compute average correct prediction rate by dividing sum of correct predictions with number of all predicionts
    df_temp['chance_correct_pred'] = df_temp.correct_pred / df_temp.ID_count

    #- compute average correct prediction rate by dividing sum of correct predictions with number of all predicionts
    df_temp['chance_correct_confl_pred'] = df_temp.nr_pred_confl / df_temp.ID_count

    #- merge with global dataframe containing IDs and geometry, and keep only those polygons occuring in test sample
    df_hit = pd.merge(df_temp, global_df, on='ID', how='left')

    #- convert to geodataframe
    gdf_hit = gpd.GeoDataFrame(df_hit, geometry=df_hit.geometry)

    return df_hit, gdf_hit

def init_out_ROC_curve():
    """initiates empty lists for range of variables needed to plot ROC-curve per simulation.

    Returns:
        lists: empty lists for variables.
    """    

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    return tprs, aucs, mean_fpr

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

def correlation_matrix(df):
    """Computes the correlation matrix for a dataframe.

    Args:
        df (dataframe): dataframe with analysed output per polygon.

    Returns:
        dataframe: dataframe containig correlation matrix.
    """    

    df = df.drop('geometry', axis=1)
    df = df.corr()
    df

    return df

def categorize_polys(gdf_hit, mode='median'):
    """Categorizes polygons depending on the computed chance of correct predictions as main category, and number of conflicts in test-dat per polygon as sub-category.
    This can help to identify polygons where model predictions can be trust more, particularly with respect to correct predictions of conflict areas.
    
    Main categories are:
    * H: chance of correct prediction higher than treshold;
    * L: chance of correct prediction lower than treshold.

    Sub-categories are:
    * HH: high chance of correct prediction with high number of conflicts;
    * HL: high chance of correct prediction with low number of conflicts;
    * LH: low chance of correct prediction with high number of conflicts;
    * LL: low chance of correct prediction with low number of conflicts.

    Args:
        gdf_hit (geo-dataframe): geo-dataframe containing model evaluation per unique polygon.
        mode (str, optional): Statistical mode used to determine categorization threshold. Defaults to 'median'.

    Raises:
        ValueError: error raised if mode is neither 'median' nor 'mean'.

    Returns:
        geo-dataframe: geo-dataframe containing polygon categorization.
    """    

    if mode == 'median':
        average_hit_median = gdf_hit.chance_correct_pred.median()
        nr_confl_median = gdf_hit.nr_test_confl.median()
    elif mode == 'mean':
        average_hit_median = gdf_hit.chance_correct_pred.mean()
        nr_confl_median = gdf_hit.nr_test_confl.mean()
    else:
        raise ValueError('specified mode not supported - use either median (default) or mean')

    gdf_hit['main_category'] = ''
    gdf_hit['main_category'].loc[gdf_hit.chance_correct_pred >= average_hit_median] = 'H'
    gdf_hit['main_category'].loc[gdf_hit.chance_correct_pred < average_hit_median] = 'L'

    gdf_hit['sub_category'] = ''
    gdf_hit['sub_category'].loc[(gdf_hit.chance_correct_pred >= average_hit_median) & 
                        (gdf_hit.nr_test_confl >= nr_confl_median)] = 'HH'
    gdf_hit['sub_category'].loc[(gdf_hit.chance_correct_pred >= average_hit_median) & 
                        (gdf_hit.nr_test_confl < nr_confl_median)] = 'HL'
    gdf_hit['sub_category'].loc[(gdf_hit.chance_correct_pred < average_hit_median) & 
                        (gdf_hit.nr_test_confl >= nr_confl_median)] = 'LH'
    gdf_hit['sub_category'].loc[(gdf_hit.chance_correct_pred < average_hit_median) & 
                        (gdf_hit.nr_test_confl < nr_confl_median)] = 'LL'

    return gdf_hit