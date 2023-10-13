import copro
import os, sys
import click
from sklearn import metrics, inspection
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.datasets import load_iris
import pickle
import matplotlib.pyplot as plt

def init_out_dict(config):
    """Initiates the main model evaluation dictionary for a range of model metric scores. 
    The scores should match the scores used in the dictioary created in 'evaluation.evaluate_prediction()'.

     Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        model: the choosen ML-model

    Returns:
        dict: empty dictionary with metrics as keys.
    """  

    if config.get('machine_learning', 'model') == 'RFClassifier':
       scores = ['Accuracy', 'Precision', 'Recall', 'F1 score', 'Cohen-Kappa score', 'Brier loss score', 'ROC AUC score', 'AP score']
         
    elif config.get('machine_learning', 'model') == 'RFRegression':
        scores = ['Mean Absolute Error', 'Mean Squared Error', 'R2 Score']  
        print ('INFO: calculated evaluation for RFRegression')               

     # initialize empty dictionary with one emtpy list per score
    out_dict = {}
    for score in scores:
        out_dict[score] = list()

    return out_dict

def evaluate_prediction_classifier(y_test, y_pred, y_prob, X_test, mdl, config):
     
    """Computes a range of model evaluation metrics and appends the resulting scores to a dictionary.
    This is done for each model execution separately.
    Output will be stored to stderr if possible.

    Args:
        y_test (list): list containing test-sample migration data.
        y_pred (list): list containing predictions.
        y_prob (array): array resulting probabilties of predictions.
        X_test (array): array containing test-sample variable values.
        mdl (model): sklearn-approach used in the simulation.
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
       click.echo('... AP score {0:0.3f}'.format(metrics.average_precision_score(y_test, y_prob[:, 1])), err=True)

    # compute value per evaluation metric depending on ML-model settings and assign to list

    eval_dict = {'Accuracy': metrics.accuracy_score(y_test, y_pred),
                 'Precision': metrics.precision_score(y_test, y_pred),
                 'Recall': metrics.recall_score(y_test, y_pred),
                 'F1 score': metrics.f1_score(y_test, y_pred),
                 'Cohen-Kappa score': metrics.cohen_kappa_score(y_test, y_pred),
                 'Brier loss score': metrics.brier_score_loss(y_test, y_prob[:, 1]),
                 'ROC AUC score': metrics.roc_auc_score(y_test, y_prob[:, 1]),
                 'AP score': metrics.average_precision_score(y_test, y_prob[:, 1]),
    }
    
    return eval_dict

def evaluate_prediction_regression(y_test, y_pred, X_test, mdl, config):
   
    """Computes a range of model evaluation metrics for a MLregression and appends the resulting scores to a dictionary.
    This is done for each model execution separately.
    Output will be stored to stderr if possible.

    Args:
        y_test (list): list containing test-sample migration data.
        y_pred (list): list containing predictions.
        X_test (array): array containing test-sample variable values.
        mdl (model): sklearn-approach used in the simulation.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        dict: dictionary with scores for each simulation
    """  
   
    if config.getboolean('general', 'verbose'):     
        click.echo("... Mean Absolute Error: {0:0.3f}".format(metrics.mean_absolute_error(y_test, y_pred)), err=True)
        click.echo("... Mean Squared Error: {0:0.3f}".format(metrics.mean_squared_error(y_test, y_pred)), err=True)
        click.echo("... R2 Score: {0:0.3f}".format(metrics.r2_score(y_test, y_pred)), err=True)
        
    # compute value per evaluation metric depending on ML-model settings and assign to list

    eval_dict = {'Mean Absolute Error': metrics.mean_absolute_error(y_test, y_pred),
                 'Mean Squared Error': metrics.mean_squared_error(y_test, y_pred),
                 'R2 Score': metrics.r2_score(y_test, y_pred),
                 }

    return eval_dict

def fill_out_dict(out_dict, eval_dict):
    """Appends the computed metric score per run to the main output dictionary.
    All metrics are initialized in init_out_dict().

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

    out_df = pd.concat([out_df, y_df], ignore_index=True)

    return out_df

def polygon_model_accuracy(df, global_df, make_proj=False):
    """Determines a range of model accuracy values for each polygon.
    Reduces dataframe with results from each simulation to values per unique polygon identifier.
    Determines the total number of predictions made per polygon as well as fraction of correct predictions made for overall and migration-only data.

    Args:
        df (dataframe): output dataframe containing results of all simulations.
        global_df (dataframe): global look-up dataframe to associate unique identifier with geometry.
        make_proj (bool, optional): whether or not this function is used to make a projection. If True, a couple of calculations are skipped as no observed data is available for projections. Defaults to 'False'.

    Returns:
        (geo-)dataframe: dataframe and geo-dataframe with data per polygon.
    """    

    #- create a dataframe containing the number of occurence per ID
    ID_count = df.ID.value_counts().to_frame().rename(columns={'count':'nr_predictions'})
    ID_count['ID'] = ID_count.index.values
    #- set index with index named ID now
    ID_count.set_index(ID_count.ID, inplace=True)
    #- remove column ID
    ID_count = ID_count.drop('ID', axis=1)

    df_count = pd.DataFrame()
    
    #- per polygon ID, compute sum of overall correct predictions and rename column name
    if not make_proj: df_count['nr_correct_predictions'] = df.correct_pred.groupby(df.ID).sum()

    #- per polygon ID, compute sum of all in-migration data and add to dataframe
    if not make_proj: df_count['nr_observations'] = df.y_test.groupby(df.ID).sum()

    #- per polygon ID, compute sum of all 1 (in migration) data points and add to dataframe  
    df_count['predicted_in_migration'] = df.y_pred.groupby(df.ID).sum()

    #- per polygon ID, compute average probability that in-migration occurs
    df_count['min_prob_1'] = pd.to_numeric(df.y_prob_1).groupby(df.ID).min()
    df_count['probability_of_in_migration'] = pd.to_numeric(df.y_prob_1).groupby(df.ID).mean()
    df_count['max_prob_1'] = pd.to_numeric(df.y_prob_1).groupby(df.ID).max()

    #- merge the two dataframes with ID as key
    df_temp = pd.merge(ID_count, df_count, on='ID')

    #- compute average correct prediction rate by dividing sum of correct predictions with number of all predicionts
    if not make_proj: df_temp['fraction_correct_predictions'] = df_temp.nr_correct_predictions / df_temp.nr_predictions

    #- compute average correct prediction rate by dividing sum of correct predictions with number of all predicionts
    df_temp['chance_of_in_migration'] = df_temp.predicted_in_migration / df_temp.nr_predictions

    #- merge with global dataframe containing IDs and geometry, and keep only those polygons occuring in test sample
    df_hit = pd.merge(df_temp, global_df, on='ID', how='left')

    #- convert to geodataframe
    gdf_hit = gpd.GeoDataFrame(df_hit, geometry=df_hit.geometry)

    return df_hit, gdf_hit

def init_out_ROC_curve():
    """Initiates empty lists for range of variables needed to plot ROC-curve per simulation.

    Returns:
        lists: empty lists for variables.
    """    

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    return tprs, aucs, mean_fpr

def save_out_ROC_curve(tprs, aucs, out_dir):
    """Saves data needed to plot mean ROC and standard deviation to csv-files. 
    They can be loaded again with pandas in a post-processing step.

    Args:
        tprs (list): list with false positive rates.
        aucs (list): list with area-under-curve values.
        out_dir (str):  path to output folder. If 'None', no output is stored.
    """    

    tprs = pd.DataFrame(tprs)
    aucs = pd.DataFrame(aucs)

    tprs.to_csv(os.path.join(out_dir, 'ROC_data_tprs.csv'), index=False, header=False)
    aucs.to_csv(os.path.join(out_dir, 'ROC_data_aucs.csv'), index=False, header=False)

    print('INFO: saving ROC data to {}'.format(os.path.join(out_dir, 'ROC_data.csv')))

def calc_correlation_matrix(df, out_dir=None):
    """Computes the correlation matrix for a dataframe. 
    The dataframe should only contain numeric values.

    Args:
        df (dataframe): dataframe with analysed output per polygon.
        out_dir (str):  path to output folder. If 'None', no output is stored. Default to 'None'.

    Returns:
        dataframe: dataframe containig correlation matrix.
    """    

    # determine correlation matrix
    df_corr = df.corr()
    
    if (out_dir != None) and isinstance(out_dir, str):
        df_corr.to_csv(os.path.join(out_dir, 'corr_matrix.csv'))

    return df_corr

def get_feature_importance(mdl, config, out_dir):
    """Determines relative importance of each feature (i.e. variable) used. Must be used after model/classifier is fit.
    Returns dataframe and saves it to csv too.

    Args:
        mdl (model): sklearn-model used in the simulation.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        out_dir (str): path to output folder. If 'None', no output is stored.

    Raises:
        Warning: raised if the chosen ML model has not built-in feature importances.

    Returns:
        dataframe: dataframe containing feature importance.
    """ 

    if config.get('machine_learning', 'model') == 'RFClassifier':

        # get feature importances
        arr = mdl.feature_importances_

        # Initialize dictionary and add importance value per feature
        dict_out = dict()
        for key, importance in zip(config.items('data'), arr):
            dict_out[key[0]] = importance

        # Convert to dataframe
        df = pd.DataFrame.from_dict(dict_out, orient='index', columns=['feature_importance'])

        # Sort the DataFrame by Permutation Importance in descending order
        feat_importance_df = df.sort_values(by='feature_importance', ascending=False)
    
        # Calculate relative permutation importances
        feat_importance_df['Relative Feature Importance'] = feat_importance_df['feature_importance'] / feat_importance_df['feature_importance'].max()

        if (out_dir != None) and isinstance(out_dir, str):
            feat_importance_df.to_csv(os.path.join(out_dir, 'feature_importance.csv'))


    elif config.get('machine_learning', 'model') == 'RFRegression':
        # Get feature importances
        arr = mdl.feature_importances_

        # Initialize dictionary and add importance value per feature
        dict_out = dict()
        for key, importance in zip(config.items('data'), arr):
            dict_out[key[0]] = importance

        # Convert to dataframe
        df = pd.DataFrame.from_dict(dict_out, orient='index', columns=['feature_importance'])

        # Sort the DataFrame by Permutation Importance in descending order
        feat_importance_df = df.sort_values(by='feature_importance', ascending=False)
    
        # Calculate relative permutation importances
        feat_importance_df['Relative Feature Importance'] = feat_importance_df['feature_importance'] / feat_importance_df['feature_importance'].max()

        if (out_dir != None) and isinstance(out_dir, str):
            feat_importance_df.to_csv(os.path.join(out_dir, 'feature_importance.csv'))

    else:
        raise Warning('WARNING: feature importance not supported for {}'.format(config.get('machine_learning', 'model')))

    return df

def get_permutation_importance(mdl, X_ft, Y, df_feat_imp, out_dir):
    """Returns a dataframe with the mean permutation importance of the features used to train a RF tree model.
    Dataframe is stored to output directory as csv-file.

    Args:
        mdl (model): sklearn-model used in the simulation.
        X_ft (array): X-array containing variable values after scaling.
        Y (array): Y-array containing in-migration data.
        df_feat_imp (dataframe): dataframe containing feature importances to align names across outputs.
        out_dir (str): path to output folder. If 'None', no output is stored.

    Returns:
        dataframe: contains mean permutation importance for each feature.
    """    
    print(Y)
    print(X_ft)
    
    result = inspection.permutation_importance(mdl, X_ft, Y, n_repeats=10, random_state=42)   

    df = pd.DataFrame(result.importances_mean, columns=['permutation_importance'], index=df_feat_imp.index.values)
    df['feature'] = df_feat_imp.index.values

    # Sort the DataFrame by Permutation Importance in descending order
    perm_importance_df = df.sort_values(by='permutation_importance', ascending=False)
    
    # Calculate relative permutation importances
    perm_importance_df['Relative Permutation Importance'] = perm_importance_df['permutation_importance'] / perm_importance_df['permutation_importance'].max()

    if (out_dir != None) and isinstance(out_dir, str):
        perm_importance_df.to_csv(os.path.join(out_dir, 'permutation_importances.csv'))

    return perm_importance_df 

def make_average_tree(config_REF, out_dir_REF, mdl):
    # Load the pickled object
    with open(os.path.join(out_dir_REF, 'mdls'), 'rb') as f:
        mdl_obj = pickle.load(f)
                  
    # Extract the individual decision trees
    individual_trees = mdl_obj.estimators_

    # Calculate the average decision tree
    average_tree = np.mean([tree.tree_ for tree in individual_trees], axis=0)
               
    # Create a figure to visualize the average tree
    fig, ax = plt.subplots(figsize=(10, 10))

    figure_filename = os.path.join(out_dir_REF, 'average_tree.png')
    fig.savefig(figure_filename)