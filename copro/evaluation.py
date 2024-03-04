import os
import click
from sklearn import metrics, inspection, ensemble
from configparser import RawConfigParser
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Union, Tuple


def init_out_dict(scores: Union[list[str], None] = None) -> dict:
    """Initiates the main model evaluatoin dictionary for a range of model metric scores.
    The scores should match the scores used in the dictioary created in 'evaluation.evaluate_prediction()'.

    Args:
        scores (list, None): list containing metric scores. If 'None', a default list is used. Defaults to 'None'.

    Returns:
        dict: empty dictionary with metrics as keys.
    """

    if scores is None:
        scores = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 score",
            "Cohen-Kappa score",
            "Brier loss score",
            "ROC AUC score",
            "AP score",
        ]

    # initialize empty dictionary with one emtpy list per score
    out_dict = {}
    for score in scores:
        out_dict[score] = []

    return out_dict


def fill_out_dict(
    out_dict: dict,
    y_test: list[int],
    y_pred: list[int],
    y_prob: list[float],
    config: RawConfigParser,
):
    """Appends the computed metric score per run to the main output dictionary.
    All metrics are initialized in init_out_dict().

    Args:
        out_dict (dict): main output dictionary.
        eval_dict (dict): dictionary containing scores per simulation.

    Returns:
        dict: dictionary with collected scores for each simulation
    """

    eval_dict = evaluate_prediction(y_test, y_pred, y_prob, config)

    for key in out_dict:
        out_dict[key].append(eval_dict[key])

    return out_dict


def polygon_model_accuracy(
    df: pd.DataFrame, global_df: pd.DataFrame, make_proj=False
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Determines a range of model accuracy values for each polygon.
    Reduces dataframe with results from each simulation to values per unique polygon identifier.
    Determines the total number of predictions made per polygon 
    as well as fraction of correct predictions made for overall and conflict-only data.

    Args:
        df (dataframe): output dataframe containing results of all simulations.
        global_df (dataframe): global look-up dataframe to associate unique identifier with geometry.
        make_proj (bool, optional): whether or not this function is used to make a projection. \
            If True, a couple of calculations are skipped as no observed data is available for projections. \
                Defaults to 'False'.

    Returns:
        (geo-)dataframe: dataframe and geo-dataframe with data per polygon.
    """

    # - create a dataframe containing the number of occurence per ID
    ID_count = df.ID.value_counts().to_frame().rename(columns={"ID": "nr_predictions"})
    # - add column containing the IDs
    ID_count["ID"] = ID_count.index.values
    # - set index with index named ID now
    ID_count.set_index(ID_count.ID, inplace=True)
    # - remove column ID
    ID_count = ID_count.drop("ID", axis=1)

    df_count = pd.DataFrame()

    # - per polygon ID, compute sum of overall correct predictions and rename column name
    if not make_proj:
        df_count["nr_correct_predictions"] = df.correct_pred.groupby(df.ID).sum()

    # - per polygon ID, compute sum of all conflict data points and add to dataframe
    if not make_proj:
        df_count["nr_observed_conflicts"] = df.y_test.groupby(df.ID).sum()

    # - per polygon ID, compute sum of all conflict data points and add to dataframe
    df_count["nr_predicted_conflicts"] = df.y_pred.groupby(df.ID).sum()

    # - per polygon ID, compute average probability that conflict occurs
    df_count["min_prob_1"] = pd.to_numeric(df.y_prob_1).groupby(df.ID).min()
    df_count["probability_of_conflict"] = (
        pd.to_numeric(df.y_prob_1).groupby(df.ID).mean()
    )
    df_count["max_prob_1"] = pd.to_numeric(df.y_prob_1).groupby(df.ID).max()

    # - merge the two dataframes with ID as key
    df_temp = pd.merge(ID_count, df_count, on="ID")

    # - compute average correct prediction rate by dividing sum of correct predictions with number of all predicionts
    if not make_proj:
        df_temp["fraction_correct_predictions"] = (
            df_temp.nr_correct_predictions / df_temp.nr_predictions
        )

    # - compute average correct prediction rate by dividing sum of correct predictions with number of all predicionts
    df_temp["chance_of_conflict"] = (
        df_temp.nr_predicted_conflicts / df_temp.nr_predictions
    )

    # - merge with global dataframe containing IDs and geometry, and keep only those polygons occuring in test sample
    df_hit = pd.merge(df_temp, global_df, on="ID", how="left")
    # #- convert to geodataframe
    gdf_hit = gpd.GeoDataFrame(df_hit, geometry=df_hit.geometry)

    return df_hit, gdf_hit


def get_feature_importance(
    clf: ensemble.RandomForestClassifier,
    config: RawConfigParser,
    out_dir: Union[str, None] = None,
) -> pd.DataFrame:
    """Determines relative importance of each feature (i.e. variable) used. Must be used after model/classifier is fit.
    Returns dataframe and saves it to csv too.

    Args:
        clf (classifier): sklearn-classifier used in the simulation.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        out_dir (str): path to output folder. If 'None', no output is stored.

    Returns:
        dataframe: dataframe containing feature importance.
    """

    if not os.path.exists(out_dir):
        raise FileNotFoundError("Output directory does not exist.")

    # get feature importances
    arr = clf.feature_importances_

    # initialize dictionary and add importance value per indicator
    dict_out = {}
    for key, x in zip(config.items("data"), range(len(arr))):
        dict_out[key[0]] = arr[x]
    dict_out["conflict_t_min_1"] = arr[-2]
    dict_out["conflict_t_min_1_nb"] = arr[-1]

    # convert to dataframe
    df = pd.DataFrame.from_dict(
        dict_out, orient="index", columns=["feature_importance"]
    )
    if out_dir is not None:
        df.to_csv(os.path.join(out_dir, "feature_importances.csv"))

    return df


def get_permutation_importance(
    clf: ensemble.RandomForestClassifier,
    X_ft: np.ndarray,
    Y: np.ndarray,
    df_feat_imp: pd.DataFrame,
    out_dir: Union[str, None] = None,
    n_repeats=10,
) -> pd.DataFrame:
    """Returns a dataframe with the mean permutation importance of the features used to train a RF tree model.
    Dataframe is stored to output directory as csv-file.

    Args:
        clf (classifier): sklearn-classifier used in the simulation.
        X_ft (array): X-array containing variable values after scaling.
        Y (array): Y-array containing conflict data.
        df_feat_imp (dataframe): dataframe containing feature importances to align names across outputs.
        out_dir (str, None): path to output folder. If 'None', no output is stored. Default to 'None'.
        n_repeats (int): number of repetitions for permutation importance. Default to 10.

    Returns:
        dataframe: contains mean permutation importance for each feature.
    """

    if not os.path.exists(out_dir):
        raise FileNotFoundError("Output directory does not exist.")

    result = inspection.permutation_importance(
        clf, X_ft, Y, n_repeats=n_repeats, random_state=42
    )
    df = pd.DataFrame(
        result.importances_mean,
        columns=["permutation_importance"],
        index=df_feat_imp.index.values,
    )
    if out_dir is not None:
        df.to_csv(os.path.join(out_dir, "permutation_importances.csv"))

    return df


def evaluate_prediction(
    y_test: list, y_pred: list, y_prob: list, config: RawConfigParser
) -> dict:
    """Computes a range of model evaluation metrics and appends the resulting scores to a dictionary.
    This is done for each model execution separately.
    Output will be stored to stderr if possible.

    Args:
        y_test (list): list containing test-sample conflict data.
        y_pred (list): list containing predictions.
        y_prob (array): array resulting probabilties of predictions.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        dict: dictionary with scores for each simulation
    """

    if config.getboolean("general", "verbose"):
        click.echo(
            "... Accuracy: {0:0.3f}".format(metrics.accuracy_score(y_test, y_pred)),
            err=True,
        )
        click.echo(
            "... Precision: {0:0.3f}".format(metrics.precision_score(y_test, y_pred)),
            err=True,
        )
        click.echo(
            "... Recall: {0:0.3f}".format(metrics.recall_score(y_test, y_pred)),
            err=True,
        )
        click.echo(
            "... F1 score: {0:0.3f}".format(metrics.f1_score(y_test, y_pred)), err=True
        )
        click.echo(
            "... Brier loss score: {0:0.3f}".format(
                metrics.brier_score_loss(y_test, y_prob[:, 1])
            ),
            err=True,
        )
        click.echo(
            "... Cohen-Kappa score: {0:0.3f}".format(
                metrics.cohen_kappa_score(y_test, y_pred)
            ),
            err=True,
        )
        click.echo(
            "... ROC AUC score {0:0.3f}".format(
                metrics.roc_auc_score(y_test, y_prob[:, 1])
            ),
            err=True,
        )
        click.echo(
            "... AP score {0:0.3f}".format(
                metrics.average_precision_score(y_test, y_prob[:, 1])
            ),
            err=True,
        )

    # compute value per evaluation metric and assign to list
    eval_dict = {
        "Accuracy": metrics.accuracy_score(y_test, y_pred),
        "Precision": metrics.precision_score(y_test, y_pred),
        "Recall": metrics.recall_score(y_test, y_pred),
        "F1 score": metrics.f1_score(y_test, y_pred),
        "Cohen-Kappa score": metrics.cohen_kappa_score(y_test, y_pred),
        "Brier loss score": metrics.brier_score_loss(y_test, y_prob[:, 1]),
        "ROC AUC score": metrics.roc_auc_score(y_test, y_prob[:, 1]),
        "AP score": metrics.average_precision_score(y_test, y_prob[:, 1]),
    }

    return eval_dict
