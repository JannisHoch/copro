import click
import pandas as pd
import numpy as np
import geopandas as gpd


def neighboring_polys(
    extent_gdf: gpd.GeoDataFrame, identifier="watprovID"
) -> pd.DataFrame:
    """For each polygon, determines its neighboring polygons.
    As result, a (n x n) look-up dataframe is obtained containing, where n is number of polygons in extent_gdf.

    Args:
        extent_gdf (geo-dataframe): geo-dataframe containing the selected polygons.
        identifier (str, optional): column name in extent_gdf to be used to identify neighbors. Defaults to 'watprovID'.

    Returns:
        dataframe: look-up dataframe containing True/False statement per polygon for all other polygons.
    """

    click.echo("Determining matrix with neighboring polygons.")
    # initialise empty dataframe
    df = pd.DataFrame()
    # go through each polygon aka water province
    for i in range(len(extent_gdf)):
        # get geometry of current polygon
        wp = extent_gdf.geometry.iloc[i]
        # check which polygons in geodataframe (i.e. all water provinces) touch the current polygon
        # also create a dataframe from result (boolean)
        # the transpose is needed to easier append
        df_temp = pd.DataFrame(
            extent_gdf.geometry.touches(wp), columns=[extent_gdf[identifier].iloc[i]]
        ).T
        # append the dataframe
        df = df.append(df_temp)

    # replace generic indices with actual water province IDs
    df.set_index(extent_gdf[identifier], inplace=True)
    # replace generic columns with actual water province IDs
    df.columns = extent_gdf[identifier].values

    return df


def find_neighbors(ID: int, neighboring_matrix: pd.DataFrame) -> np.ndarray:
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
