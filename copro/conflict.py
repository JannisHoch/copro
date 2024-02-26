from copro import utils
from configparser import RawConfigParser
from pathlib import Path
from typing import Union, Literal
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import click
import warnings


def conflict_in_year_bool(
    config: RawConfigParser,
    conflict_gdf: gpd.GeoDataFrame,
    extent_gdf: gpd.GeoDataFrame,
    sim_year: int,
    out_dir: click.Path,
) -> list:
    """Creates a list for each timestep with boolean information whether a conflict took place in a polygon or not.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        conflict_gdf (geodataframe): geo-dataframe containing georeferenced information of conflict.
        extent_gdf (geodataframe): geo-dataframe containing one or more polygons \
            with geometry information for which values are extracted.
        sim_year (int): year for which data is extracted.
        out_dir (str): path to output folder. If 'None', no output is stored.

    Returns:
        list: list containing 0/1 per polygon depending on conflict occurence.
    """

    click.echo(f"Checking for conflict events which occured in {sim_year}.")
    # select the entries which occured in this year
    temp_sel_year = conflict_gdf.loc[conflict_gdf.year == sim_year]
    if temp_sel_year.empty:
        warnings.warn(
            f"No conflicts were found in sampled conflict data set for year {sim_year}."
        )

    # merge the dataframes with polygons and conflict information, creating a sub-set of polygons/regions
    data_merged = gpd.sjoin(temp_sel_year, extent_gdf)

    # determine the aggregated amount of fatalities in one region (e.g. water province)
    fatalities_per_poly = (
        data_merged["best"]
        .groupby(data_merged["watprovID"])
        .sum()
        .to_frame()
        .rename(columns={"best": "total_fatalities"})
    )

    out_dir = os.path.join(out_dir, "files")
    Path.mkdir(Path(out_dir), exist_ok=True)

    if sim_year == config.getint("settings", "y_end"):
        _store_boolean_conflict_data_to_csv(
            fatalities_per_poly, config, extent_gdf, sim_year, out_dir
        )

    # loop through all regions and check if exists in sub-set
    # if so, this means that there was conflict and thus assign value 1
    list_out = []
    for i, _ in extent_gdf.iterrows():
        i_poly = extent_gdf.iloc[i]["watprovID"]
        if i_poly in fatalities_per_poly.index.values:
            list_out.append(1)
        else:
            list_out.append(0)

    return list_out


def conflict_in_previous_year_bool(
    conflict_gdf: gpd.GeoDataFrame,
    extent_gdf: gpd.GeoDataFrame,
    sim_year: int,
    check_neighbors: bool = False,
    neighboring_matrix: Union[None, pd.DataFrame] = None,
) -> list:
    """_summary_

    Args:
        conflict_gdf (gpd.GeoDataFrame): _description_
        extent_gdf (gpd.GeoDataFrame): _description_
        sim_year (int): _description_
        check_neighbors (bool, optional): _description_. Defaults to False.
        neighboring_matrix (Union[None, pd.DataFrame], optional): _description_. Defaults to None.

    Returns:
        list: _description_
    """

    if check_neighbors:
        click.echo("Checking for conflicts in neighboring polygons at t-1")
    else:
        click.echo("Checking for conflict event in polygon at t-1")

    # get conflicts at t-1
    temp_sel_year = conflict_gdf.loc[conflict_gdf.year == sim_year - 1]
    if temp_sel_year.empty:
        warnings.warn(
            f"No conflicts were found in sampled conflict data set for year {sim_year - 1}."
        )

    # merge the dataframes with polygons and conflict information, creating a sub-set of polygons/regions
    data_merged = gpd.sjoin(temp_sel_year, extent_gdf)

    conflicts_per_poly = (
        data_merged.id.groupby(data_merged["watprovID"])
        .count()
        .to_frame()
        .rename(columns={"id": "conflict_count"})
    )

    # loop through all polygons
    list_out = []
    for i in range(len(extent_gdf)):
        i_poly = extent_gdf.watprovID.iloc[i]
        # check if polygon is in list with conflict polygons
        if i_poly in conflicts_per_poly.index.values:
            # if so, check if neighboring polygons contain conflict and assign boolean value
            if check_neighbors:
                val = calc_conflicts_nb(i_poly, neighboring_matrix, conflicts_per_poly)
                # append resulting value
                list_out.append(val)
            # if not, assign 1 directly
            else:
                list_out.append(1)
        else:
            # if polygon not in list with conflict polygons, assign 0
            list_out.append(0)

    return list_out


def read_projected_conflict(
    extent_gdf: gpd.GeoDataFrame,
    bool_conflict: pd.DataFrame,
    check_neighbors=False,
    neighboring_matrix=None,
) -> list:
    """Creates a list for each timestep with boolean information 
    whether a conflict took place in a polygon or not.
    Input conflict data (`bool_conflict`) must contain an index with IDs 
    corresponding with the `watprovID` values of extent_gdf.
    Optionally, the algorithm can be extended to the neighboring polygons.

    Args:
        extent_gdf (gpd.GeoDataFrame): geo-dataframe containing one or more polygons \
            with geometry information for which values are extracted.
        bool_conflict (pd.DataFrame): dataframe with boolean values (1) for each polygon with conflict.
        check_neighbors (bool, optional): whether or not to check for conflict in neighboring polygons. \
            Defaults to `False`.
        neighboring_matrix (pd.DataFrame, optional): look-up dataframe listing all neighboring polygons. \
            Defaults to `None`.

    Returns:
        list: 1 and 0 values for each polygon with conflict respectively without conflict. \
            If `check_neighbors=True`, then 1 if neighboring polygon contains conflict and 0 is not.
    """

    # loop through all polygons and check if exists in sub-set
    list_out = []
    for i in range(len(extent_gdf)):
        i_poly = extent_gdf.watprovID.iloc[i]
        if i_poly in bool_conflict.index.values:
            if check_neighbors:
                # determine log-scaled number of conflict events in neighboring polygons
                val = calc_conflicts_nb(i_poly, neighboring_matrix, bool_conflict)
                # append resulting value
                list_out.append(val)
            else:
                list_out.append(1)
        else:
            # if polygon not in list with conflict polygons, assign 0
            list_out.append(0)

    return list_out


def calc_conflicts_nb(
    i_poly: int, neighboring_matrix: pd.DataFrame, conflicts_per_poly: pd.DataFrame
) -> Literal[0, 1]:
    """Determines whether in the neighbouring polygons of a polygon i_poly conflict took place.
    If so, a value 1 is returned, otherwise 0.

    Args:
        i_poly (int): ID number of polygon under consideration.
        neighboring_matrix (pd.DataFrame): look-up dataframe listing all neighboring polygons.
        conflicts_per_poly (pd.DataFrame): dataframe with conflict data per polygon.

    Returns:
        Literal: 1 if conflict took place in neighboring polygon, 0 if not.
    """

    # find neighbors of this polygon
    nb = _find_neighbors(i_poly, neighboring_matrix)

    # initiate list
    nb_count = []
    # loop through neighbors
    for k in nb:
        # check if there was conflict at t-1
        if k in conflicts_per_poly.index.values:
            nb_count.append(1)
    # if more than one neighboring polygon has conflict, return 0
    if np.sum(nb_count) > 0:
        val = 1
    # otherwise, return 0
    else:
        val = 0

    return val


def check_for_correct_prediction(
    X_test_ID: np.ndarray,
    X_test_geom: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob_0: np.ndarray,
    y_prob_1: np.ndarray,
) -> pd.DataFrame:
    """Stacks together the arrays with unique identifier, geometry, test data, and predicted data into a dataframe.
    Contains therefore only the data points used in the test-sample, not in the training-sample.
    Additionally computes whether a correct prediction was made.

    Args:
        X_test_ID (np.ndarray): _description_
        X_test_geom (np.ndarray): _description_
        y_test (np.ndarray): _description_
        y_pred (np.ndarray): _description_
        y_prob_0 (np.ndarray): _description_
        y_prob_1 (np.ndarray): _description_

    Returns:
        pd.DataFrame: _description_
    """

    # stack separate columns horizontally
    arr = np.column_stack((X_test_ID, X_test_geom, y_test, y_pred, y_prob_0, y_prob_1))
    # convert array to dataframe
    df = pd.DataFrame(
        arr, columns=["ID", "geometry", "y_test", "y_pred", "y_prob_0", "y_prob_1"]
    )
    # compute whether a prediction is correct
    # if so, assign 1; otherwise, assign 0
    df["correct_pred"] = np.where(df["y_test"] == df["y_pred"], 1, 0)

    return df


def _store_boolean_conflict_data_to_csv(
    fatalities_per_poly: pd.DataFrame,
    extent_gdf: gpd.GeoDataFrame,
    sim_year: int,
    out_dir: click.Path,
):
    """Stores boolean conflict data to csv-file at the end of reference period.
    Used as initial conditions for projections from there.

    Args:
        fatalities_per_poly (pd.DataFrame): Fatalities per polygon in `sim_year`.
        extent_gdf (gpd.GeoDataFrame): All polygons considered in analysis, also those w/o conflict.
        sim_year (int): Simulation year for which data is stored.
        out_dir (click.Path): Path to output folder.
    """

    # get a 1 for each polygon where there was conflict
    bool_per_poly = fatalities_per_poly / fatalities_per_poly
    # change column name and dtype
    bool_per_poly = bool_per_poly.rename(
        columns={"total_fatalities": "bool_conflict"}
    ).astype(int)
    # change index name to fit global_df
    bool_per_poly.index = bool_per_poly.index.rename("ID")
    # get list of all polygon IDs with their geometry information
    global_df = utils.get_ID_geometry_lookup(extent_gdf)
    # merge the boolean info with geometry
    # for all polygons without conflict, set a 0
    click.echo(
        f"Storing boolean conflict map of year {sim_year} \
            to file {os.path.join(out_dir, f'conflicts_in_{sim_year}.csv')}"
    )

    data_stored = pd.merge(bool_per_poly, global_df, on="ID", how="right").dropna()
    data_stored.index = data_stored.index.rename("watprovID")
    data_stored = data_stored.drop("geometry", axis=1)
    data_stored = data_stored.astype(int)
    data_stored.to_csv(os.path.join(out_dir, f"conflicts_in_{sim_year}.csv"))


def _find_neighbors(ID: int, neighboring_matrix: pd.DataFrame) -> np.ndarray:
    """Filters all polygons which are actually neighbors to given polygon.

    Args:
        ID (int): ID of specific polygon under consideration.
        neighboring_matrix (pd.DataFrame): output from neighboring_polys().

    Returns:
        np.ndarray: IDs of all polygons that are actual neighbors.
    """

    # locaties entry for polygon under consideration
    neighbours = neighboring_matrix.loc[neighboring_matrix.index == ID].T

    # filters all actual neighbors defined as neighboring polygons with True statement
    actual_neighbours = neighbours.loc[  # noqa: C0121
        neighbours[ID] == True  # noqa: C0121
    ].index.values  # noqa: C0121

    return actual_neighbours
