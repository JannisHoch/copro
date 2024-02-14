import geopandas as gpd
import pandas as pd
import os
from configparser import RawConfigParser
from datetime import date
import click
from copro import __version__, __author__, __email__


def get_conflict_geodataframe(
    config: RawConfigParser,
    root_dir: click.Path,
    longitude="longitude",
    latitude="latitude",
    crs="EPSG:4326",
) -> gpd.GeoDataFrame:
    """Converts a csv-file containing geo-referenced conflict data to a geodataframe.

    Args:
        config (RawConfigParser): object containing the parsed configuration-settings of the model.
        root_dir (Path): path to location of cfg-file.
        longitude (str, optional): column name with longitude coordinates. Defaults to 'longitude'.
        latitude (str, optional): column name with latitude coordinates. Defaults to 'latitude'.
        crs (str, optional): coordinate system to be used for georeferencing. Defaults to 'EPSG:4326'.

    Returns:
        geo-dataframe: geo-referenced conflict data.
    """

    # get path to file containing data
    conflict_fo = os.path.join(
        root_dir,
        config.get("general", "input_dir"),
        config.get("conflict", "conflict_file"),
    )

    # read file to pandas dataframe
    click.echo(f"Reading {conflict_fo} file and converting to geodataframe.")
    df = pd.read_csv(conflict_fo)

    # convert dataframe to geo-dataframe
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[longitude], df[latitude]), crs=crs
    )

    return gdf


def get_ID_geometry_lookup(
    gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:  # get_ID_geometry_lookup
    """Retrieves unique ID and geometry information from geo-dataframe for a global look-up dataframe.
    The IDs currently supported are 'name' or 'watprovID'.

    Args:
        gdf (gpd.GeoDataFrame): containing all polygons used in the model.

    Returns:
        pd.DataFrame: look-up dataframe associated ID with geometry
    """

    # stack identifier and geometry of all polygons
    # NOTE: columnn 'watprovID' is hardcoded here
    df = pd.DataFrame(
        index=gdf["watprovID"].to_list(),
        data=gdf["geometry"].to_list(),
        columns=["geometry"],
    )

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
