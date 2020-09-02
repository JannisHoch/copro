import os, sys
from sklearn import metrics
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

def init_out_dict():

    scores = ['Accuracy', 'Precision', 'Recall', 'F1 score', 'Cohen-Kappa score', 'Brier loss score', 'ROC AUC score']

    out_dict = {}
    for score in scores:
        out_dict[score] = list()

    return out_dict

def init_out_df():

    return pd.DataFrame()

def fill_out_dict(out_dict, eval_dict):

    for key in out_dict:
        out_dict[key].append(eval_dict[key])

    return out_dict

def fill_out_df(out_df, y_df):

    out_df = out_df.append(y_df, ignore_index=True)

    return out_df

def evaluate_prediction(y_test, y_pred, y_prob, X_test, clf, out_dir):
    """[summary]

    Args:
        y_test ([type]): [description]
        y_pred ([type]): [description]
        y_prob ([type]): [description]
    """    

    print("Accuracy: {0:0.3f}".format(metrics.accuracy_score(y_test, y_pred)))
    print("Precision: {0:0.3f}".format(metrics.precision_score(y_test, y_pred)))
    print("Recall: {0:0.3f}".format(metrics.recall_score(y_test, y_pred)))
    print('F1 score: {0:0.3f}'.format(metrics.f1_score(y_test, y_pred)))
    print('Brier loss score: {0:0.3f}'.format(metrics.brier_score_loss(y_test, y_prob[:, 1])))
    print('Cohen-Kappa score: {0:0.3f}'.format(metrics.cohen_kappa_score(y_test, y_pred)))
    print('ROC AUC score {0:0.3f}'.format(metrics.roc_auc_score(y_test, y_pred)))
    print('')

    print(metrics.classification_report(y_test, y_pred))
    print('')

    fig, ax = plt.subplots(1, 1, figsize=(20,10))
    disp = metrics.plot_precision_recall_curve(clf, X_test, y_test, ax=ax)
    plt.savefig(os.path.join(out_dir, 'precision_recall_curve.png'), dpi=300)
    plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    metrics.plot_confusion_matrix(clf, X_test, y_test, ax=ax)
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(20,10))
    metrics.plot_roc_curve(clf, X_test, y_test, ax=ax)
    plt.savefig(os.path.join(out_dir, 'ROC_curve.png'), dpi=300)               
    plt.close()

    eval_dict = {'Accuracy': metrics.accuracy_score(y_test, y_pred),
                 'Precision': metrics.precision_score(y_test, y_pred),
                 'Recall': metrics.recall_score(y_test, y_pred),
                 'F1 score': metrics.f1_score(y_test, y_pred),
                 'Cohen-Kappa score': metrics.cohen_kappa_score(y_test, y_pred),
                 'Brier loss score': metrics.brier_score_loss(y_test, y_prob[:, 1]),
                 'ROC AUC score': metrics.roc_auc_score(y_test, y_pred),
                }

    return eval_dict

def get_average_hit(df, global_df):

    #- create a dataframe containing the number of occurence per ID
    ID_count = df.ID.value_counts().to_frame().rename(columns={'ID':'ID_count'})
    #- add column containing the IDs
    ID_count['ID'] = ID_count.index.values
    #- set index with index named ID now
    ID_count.set_index(ID_count.ID, inplace=True)
    #- remove column ID
    ID_count = ID_count.drop('ID', axis=1)

    #- per polygon ID, compute sum of overall correct predictions and rename column name
    hit_count = df.overall_hit.groupby(df.ID).sum().to_frame().rename(columns={'overall_hit':'total_hits'})

    #- per polygon ID, compute sum of all conflict data points and add to dataframe
    hit_count['nr_of_test_confl'] = df.y_test.groupby(df.ID).sum()

    #- merge the two dataframes with ID as key
    df_temp = pd.merge(ID_count, hit_count, on='ID')

    #- compute average correct prediction rate by dividing sum of correct predictions with number of all predicionts
    df_temp['average_hit'] = df_temp.total_hits / df_temp.ID_count

    #- merge with global dataframe containing IDs and geometry, and keep only those polygons occuring in test sample
    df_hit = pd.merge(df_temp, global_df, on='ID', how='left')

    #- convert to geodataframe
    gdf_hit = gpd.GeoDataFrame(df_hit, geometry=df_hit.geometry)

    return df_hit, gdf_hit

def plot_ROC_n_times(clf, X, Y, trps, aucs):

    fig, ax = plt.subplots()
    clf.fit(X, y)
    viz = metrics.plot_roc_curve(clf, X, Y, alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)


    return ax, trps, aucs