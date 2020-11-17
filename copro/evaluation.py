import os, sys
import click
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
    Output will be stored to stderr if possible.

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
        click.echo("... Accuracy: {0:0.3f}".format(metrics.accuracy_score(y_test, y_pred)), err=True)
        click.echo("... Precision: {0:0.3f}".format(metrics.precision_score(y_test, y_pred)), err=True)
        click.echo("... Recall: {0:0.3f}".format(metrics.recall_score(y_test, y_pred)), err=True)
        click.echo('... F1 score: {0:0.3f}'.format(metrics.f1_score(y_test, y_pred)), err=True)
        click.echo('... Brier loss score: {0:0.3f}'.format(metrics.brier_score_loss(y_test, y_prob[:, 1])), err=True)
        click.echo('... Cohen-Kappa score: {0:0.3f}'.format(metrics.cohen_kappa_score(y_test, y_pred)), err=True)
        click.echo('... ROC AUC score {0:0.3f}'.format(metrics.roc_auc_score(y_test, y_prob[:, 1])), err=True)

    eval_dict = {'Accuracy': metrics.accuracy_score(y_test, y_pred),
                 'Precision': metrics.precision_score(y_test, y_pred),
                 'Recall': metrics.recall_score(y_test, y_pred),
                 'F1 score': metrics.f1_score(y_test, y_pred),
                 'Cohen-Kappa score': metrics.cohen_kappa_score(y_test, y_pred),
                 'Brier loss score': metrics.brier_score_loss(y_test, y_prob[:, 1]),
                 'ROC AUC score': metrics.roc_auc_score(y_test, y_prob[:, 1]),
                }

    # out = pd.DataFrame().from_dict(eval_dict)
    # out.to_csv(os.path.join(out_dir, 'eval_dict.csv'))

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

def polygon_model_accuracy(df, global_df, out_dir, make_proj=False):
    """Determines a range of model accuracy values for each polygon.
    Reduces dataframe with results from each simulation to values per unique polygon identifier.
    Determines the total number of predictions made per polygon as well as fraction of correct predictions made for overall and conflict-only data.

    Args:
        df (dataframe): output dataframe containing results of all simulations.
        global_df (dataframe): global look-up dataframe to associate unique identifier with geometry.
        out_dir (str): path to output folder. If 'None', no output is stored.
        make_proj (bool, optional): whether or not this function is used to make a projection. If False, a couple of calculations are skipped. Defaults to 'False'.

    Returns:
        (geo-)dataframe: dataframe and geo-dataframe with data per polygon.
    """    

    #- create a dataframe containing the number of occurence per ID
    ID_count = df.ID.value_counts().to_frame().rename(columns={'ID':'nr_predictions'})
    #- add column containing the IDs
    ID_count['ID'] = ID_count.index.values
    #- set index with index named ID now
    ID_count.set_index(ID_count.ID, inplace=True)
    #- remove column ID
    ID_count = ID_count.drop('ID', axis=1)

    df_count = pd.DataFrame()
    
    #- per polygon ID, compute sum of overall correct predictions and rename column name
    if not make_proj: df_count['nr_correct_predictions'] = df.correct_pred.groupby(df.ID).sum()

    #- per polygon ID, compute sum of all conflict data points and add to dataframe
    if not make_proj: df_count['nr_observed_conflicts'] = df.y_test.groupby(df.ID).sum()

    #- per polygon ID, compute sum of all conflict data points and add to dataframe
    df_count['nr_predicted_conflicts'] = df.y_pred.groupby(df.ID).sum()

    #- merge the two dataframes with ID as key
    df_temp = pd.merge(ID_count, df_count, on='ID')

    #- compute average correct prediction rate by dividing sum of correct predictions with number of all predicionts
    if not make_proj: df_temp['fraction_correct_predictions'] = df_temp.nr_correct_predictions / df_temp.nr_predictions

    #- compute average correct prediction rate by dividing sum of correct predictions with number of all predicionts
    df_temp['fraction_correct_conflict_predictions'] = df_temp.nr_predicted_conflicts / df_temp.nr_predictions

    #- merge with global dataframe containing IDs and geometry, and keep only those polygons occuring in test sample
    df_hit = pd.merge(df_temp, global_df, on='ID', how='left')

    #- convert to geodataframe
    gdf_hit = gpd.GeoDataFrame(df_hit, geometry=df_hit.geometry)

    if (out_dir != None) and isinstance(out_dir, str):
        gdf_hit.to_file(os.path.join(out_dir, 'output_per_polygon.shp'), crs='EPSG:4326')

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

def save_out_ROC_curve():

    return

def calc_correlation_matrix(df):
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

def categorize_polys(gdf_hit, category='sub', mode='median'):
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
        category (str, optional): Which categories to define, either main or sub. Defaults to 'sub'.
        mode (str, optional): Statistical mode used to determine categorization threshold. Defaults to 'median'.

    Raises:
        ValueError: error raised if mode is neither 'median' nor 'mean'.

    Returns:
        geo-dataframe: geo-dataframe containing polygon categorization.
    """    

    if mode == 'median':
        average_hit_median = gdf_hit.fraction_correct_predictions.median()
        nr_confl_median = gdf_hit.nr_observed_conflicts.median()
    elif mode == 'mean':
        average_hit_median = gdf_hit.fraction_correct_predictions.mean()
        nr_confl_median = gdf_hit.nr_observed_conflicts.mean()
    else:
        raise ValueError('specified mode not supported - use either median (default) or mean')

    gdf_hit['category'] = ''

    if category == 'main':
        gdf_hit['category'].loc[gdf_hit.fraction_correct_predictions >= average_hit_median] = 'H'
        gdf_hit['category'].loc[gdf_hit.fraction_correct_predictions < average_hit_median] = 'L'

    if category == 'sub':
        gdf_hit['category'].loc[(gdf_hit.fraction_correct_predictions >= average_hit_median) & 
                            (gdf_hit.nr_observed_conflicts >= nr_confl_median)] = 'HH'
        gdf_hit['category'].loc[(gdf_hit.fraction_correct_predictions >= average_hit_median) & 
                            (gdf_hit.nr_observed_conflicts < nr_confl_median)] = 'HL'
        gdf_hit['category'].loc[(gdf_hit.fraction_correct_predictions < average_hit_median) & 
                            (gdf_hit.nr_observed_conflicts >= nr_confl_median)] = 'LH'
        gdf_hit['category'].loc[(gdf_hit.fraction_correct_predictions < average_hit_median) & 
                            (gdf_hit.nr_observed_conflicts < nr_confl_median)] = 'LL'

    return gdf_hit

def calc_kFold_polygon_analysis(y_df, global_df, out_dir, k=10):
    """Determines the mean, median, and standard deviation of correct chance of prediction (CCP) for k parts of the overall output dataframe.
    Instead of evaluating the overall output dataframe at once, this can give a better feeling of the variation in CCP between model repetitions.

    Args:
        y_df (dataframe): output dataframe containing results of all simulations.
        global_df (dataframe): global look-up dataframe to associate unique identifier with geometry.
        out_dir (str): path to output folder. If 'None', no output is stored.
        k (int, optional): number of chunks in which y_df will be split. Defaults to 10.

    Returns:
        geodataframe: geodataframe containing mean, median, and standard deviation per polygon.
    """    

    ks = np.array_split(y_df, k)

    df = pd.DataFrame()

    for i in range(len(ks)):

        ks_i = ks[i]

        df_hit, gdf_hit = polygon_model_accuracy(ks_i, global_df, out_dir=None)

        temp_df = pd.DataFrame(data=pd.concat([df_hit.fraction_correct_predictions], axis=1))

        df = pd.concat([df, temp_df], axis=1)

    df['mean_CCP'] = round(df.mean(axis=1),2)
    df['median_CCP'] = round(df.median(axis=1),2)
    df['std_CCP'] = round(df.std(axis=1), 2)

    df = pd.merge(df, global_df, on='ID')

    df = df.drop(columns=['fraction_correct_predictions'])

    gdf = gpd.GeoDataFrame(df, geometry=df.geometry)

    if (out_dir != None) and isinstance(out_dir, str):
        gdf.to_file(os.path.join(out_dir, 'output_kFoldAnalysis_per_polygon.shp'), crs='EPSG:4326')

    return gdf

def get_feature_importance(clf, config, out_dir):
    """Determines relative importance of each feature (i.e. variable) used. Must be used after model/classifier is fit.
    Returns dataframe and saves it to csv too.

    Args:
        clf (classifier): sklearn-classifier used in the simulation.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        out_dir (str): path to output folder. If 'None', no output is stored.

    Returns:
        dataframe: dataframe containing feature importance.
    """ 

    if config.get('machine_learning', 'model') == 'RFClassifier':
        arr = clf.feature_importances_
    else:
        arr = np.zeros(len(config.items('data')))
        raise Warning('WARNING: feature importance not supported for this kind of ML model')

    dict_out = dict()
    for key, x in zip(config.items('data'), range(len(arr))):
        dict_out[key[0]] = arr[x]

    df = pd.DataFrame.from_dict(dict_out, orient='index', columns=['feature_importance'])

    if (out_dir != None) and isinstance(out_dir, str):
        df.to_csv(os.path.join(out_dir, 'feature_importances.csv'))

    return df