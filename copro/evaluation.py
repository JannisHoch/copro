from sklearn import metrics
import pandas as pd
import geopandas as gpd
from typing import Union


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
    eval_dict: dict,
):
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


def polygon_model_accuracy(
    df: pd.DataFrame, global_df: pd.DataFrame, make_proj=False
) -> gpd.GeoDataFrame:
    """Determines a range of model accuracy values for each polygon.
    Reduces dataframe with results from each simulation to values per unique polygon identifier.
    Determines the total number of predictions made per polygon 
    as well as fraction of correct predictions made for overall and conflict-only data.

    Args:
        df (pd.DataFrame): output dataframe containing results of all simulations.
        global_df (pd.DataFrame): global look-up dataframe to associate unique identifier with geometry.
        make_proj (bool, optional): whether or not this function is used to make a projection. \
            If True, a couple of calculations are skipped as no observed data is available for projections. \
                Defaults to 'False'.

    Returns:
        gpd.GeoDataFrame: model accuracy data per polygon.
    """

    # - create a dataframe containing the number of occurence per ID
    ID_count = (
        df.ID.value_counts().to_frame().rename(columns={"count": "nr_predictions"})
    )

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

    return gdf_hit


def evaluate_prediction(y_test: list, y_pred: list, y_prob: list) -> dict:
    """Computes a range of model evaluation metrics and appends the resulting scores to a dictionary.
    This is done for each model execution separately.
    Output will be stored to stderr if possible.

    Args:
        y_test (list): list containing test-sample conflict data.
        y_pred (list): list containing predictions.
        y_prob (array): array resulting probabilties of predictions.

    Returns:
        dict: dictionary with scores for each simulation
    """

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
