import geopandas as gpd
import pandas as pd
import numpy as np
import os
from datetime import date
import click
from copro import __version__, __author__, __email__


def get_conflict_geodataframe(
    config: dict,
    root_dir: click.Path,
    longitude="longitude",
    latitude="latitude",
    crs="EPSG:4326",
) -> gpd.GeoDataFrame:
    """Converts a csv-file containing geo-referenced conflict data to a geodataframe.

    Args:
        config (dict): Parsed configuration-settings of the model.
        root_dir (Path): path to location of cfg-file.
        longitude (str, optional): column name with longitude coordinates. Defaults to 'longitude'.
        latitude (str, optional): column name with latitude coordinates. Defaults to 'latitude'.
        crs (str, optional): coordinate system to be used for georeferencing. Defaults to 'EPSG:4326'.

    Returns:
        gpd.GeoDataFrame: geo-referenced conflict data.
    """

    # get path to file containing data
    conflict_fo = os.path.join(
        root_dir,
        config["general"]["input_dir"],
        config["data"]["conflict"]["path"],
    )

    # read file to pandas dataframe
    click.echo(f"Reading {conflict_fo} file and converting to geodataframe.")
    df = pd.read_csv(conflict_fo)

    # convert dataframe to geo-dataframe
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[longitude], df[latitude]), crs=crs
    )

    return gdf


def get_poly_ID(extent_gdf: gpd.GeoDataFrame, identifier="watprovID") -> list:
    """Extracts and returns a list with unique identifiers for each polygon used in the model.

    Args:
        extent_gdf (gpd.GeoDataFrame): all polygons considered in model.
        identifier (str, optional): unique polygon identifier column in `extent_gdf`. Defaults to 'watprovID'.

    Returns:
        list: list with ID of each polygons.
    """

    # initiatie empty list
    list_ID = []
    # loop through all polygons
    for i in range(len(extent_gdf)):
        # append geometry of each polygon to list
        list_ID.append(extent_gdf.iloc[i][identifier])

    return list_ID


def get_poly_geometry(extent_gdf: gpd.GeoDataFrame) -> list:
    """Extracts geometry information for each polygon from geodataframe and saves to list.
    The geometry column in geodataframe must be named `geometry`.

    Args:
        extent_gdf (gpd.GeoDataFrame): all polygons considered in model.

    Returns:
        list: list with geometry of each polygons.
    """

    # initiatie empty list
    list_geometry = []
    # loop through all polygons
    for i in range(len(extent_gdf)):
        # append geometry of each polygon to list
        list_geometry.append(extent_gdf.iloc[i]["geometry"])

    return list_geometry


def get_ID_geometry_lookup(
    gdf: gpd.GeoDataFrame,
    identifier="watprovID",
) -> pd.DataFrame:
    """Retrieves unique ID and geometry information from geo-dataframe for a global look-up dataframe.
    The IDs currently supported are 'name' or 'watprovID'.

    Args:
        gdf (gpd.GeoDataFrame): containing all polygons used in the model.
        identifier (str, optional): column name in `gdf` to be used as unique identifier. Defaults to 'watprovID'.

    Returns:
        pd.DataFrame: look-up dataframe associated ID with geometry
    """

    # stack identifier and geometry of all polygons
    arr = np.column_stack((gdf[identifier].to_numpy(), gdf.geometry.to_numpy()))

    # convert to dataframe
    df = pd.DataFrame(data=arr, columns=["ID", "geometry"])

    # use column ID as index
    df.set_index(df.ID, inplace=True)
    df = df.drop("ID", axis=1)

    return df


def print_model_info():
    """click.echos a header with main model information."""

    click.echo("")
    click.echo(
        click.style("#### CoPro version {} ####".format(__version__), fg="yellow")
    )
    click.echo(
        click.style(
            "#### For information about the model, please visit https://copro.readthedocs.io/ ####",
            fg="yellow",
        )
    )
    click.echo(
        click.style(
            "#### Copyright (2020-{}): {} ####".format(date.today().year, __author__),
            fg="yellow",
        )
    )
    click.echo(click.style("#### Contact via: {} ####".format(__email__), fg="yellow"))
    click.echo(
        click.style(
            "#### The model can be used and shared under the MIT license ####"
            + os.linesep,
            fg="yellow",
        )
    )
