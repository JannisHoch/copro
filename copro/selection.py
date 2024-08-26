import geopandas as gpd
import pandas as pd
import os
from copro import utils
import click
from typing import Tuple
import warnings


def select(
    config: dict, out_dir: click.Path, root_dir: click.Path
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    """Main function performing the selection procedure.
    First, selects only conflicts matching specified properties.
    Second, clips the conflict data to a specified spatial extent.
    Third, retrieves the geometry of all polygons in the spatial extent and assigns IDs.

    Args:
        config (dict): Parsed configuration-settings of the model.
        out_dir (Path): path to output folder.
        root_dir (Path): path to location of cfg-file for reference run.

    Returns:
        gpd.GeoDataFrame: remaining conflict data after selection process.
        gpd.GeoDataFrame: remaining polygons after selection process.
        pd.DataFrame: global look-up dataframe linking polygon ID with geometry information.
    """

    # get the conflict data
    conflict_gdf = utils.get_conflict_geodataframe(config, root_dir)

    # filter based on conflict properties
    conflict_gdf = _filter_conflict_properties(conflict_gdf, config)

    # clip conflicts to a spatial extent defined as polygons
    conflict_gdf, extent_gdf = _clip_to_extent(conflict_gdf, config, root_dir)

    # get a dataframe containing the ID and geometry of all polygons
    global_df = utils.get_ID_geometry_lookup(extent_gdf)

    # save conflict data and polygon to shp-file
    conflict_gdf.to_file(
        os.path.join(out_dir, "selected_conflicts.geojson"),
        driver="GeoJSON",
        crs="EPSG:4326",
    )
    extent_gdf.to_file(
        os.path.join(out_dir, "selected_polygons.geojson"),
        driver="GeoJSON",
        crs="EPSG:4326",
    )

    return conflict_gdf, extent_gdf, global_df


def _filter_conflict_properties(
    gdf: gpd.GeoDataFrame, config: dict
) -> gpd.GeoDataFrame:
    """Filters conflict database according to certain treshold options.
    These options are 'values', 'vmin' and 'vmax'.
    These options and the conflict properties to which they are applied
    need to be specified in the YAML-file.

    Args:
        gdf (gpd.GeoDataFrame): Geodataframe containing entries with conflicts.
        config (dict): Parsed configuration-settings of the model.

    Returns:
        gpd.GeoDataFrame: geo-dataframe containing filtered entries.
    """

    gdf = gdf[
        (gdf.year >= config["general"]["y_start"])
        & (gdf.year <= config["general"]["y_end"])
    ]

    # if not thresholding options are found, return the original dataframe
    if "thresholds" not in config["data"]["conflict"]:
        click.echo("No thresholding options found in configuration file.")
        return gdf

    # otherwise, go through all variables for which tresholding is specified
    for key, value in config["data"]["conflict"]["thresholds"].items():

        # if variable is not found in the dataframe, skip it
        if key not in gdf.columns:
            warnings.warn(
                f"{key} is not found in geodataframe columns, thresholding be skipped."
            )
        # otherwise, check which option is specified and apply it
        else:
            click.echo(f"Tresholding conflict data on {key}.")
            for v, k in value.items():
                if v == "values":
                    click.echo(f"Selecting datapoints with values {k}.")
                    gdf = gdf[gdf[key].isin(k)]
                elif v == "vmin":
                    click.echo(f"Selecting datapoints greater or equal to {k}.")
                    gdf = gdf[gdf[key] >= k]
                elif v == "vmax":
                    click.echo(f"Selecting datapoints less or equal to {k}.")
                    gdf = gdf[gdf[key] <= k]
                else:
                    warnings.warn(
                        f"{v} is not a recognized tresholding option - use 'values', 'vmin' or 'vmax'."
                    )

    return gdf


def _clip_to_extent(
    conflict_gdf: gpd.GeoDataFrame, config: dict, root_dir: click.Path
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """As the original conflict data has global extent, this function clips the database
    to those entries which have occured on a specified continent.

    Args:
        conflict_gdf (gpd.GeoDataFrame): Geodataframe containing entries with conflicts.
        config (dict): Parsed configuration-settings of the model.
        root_dir (str): Path to location of cfg-file.

    Returns:
        gpd.GeoDataFrame: Geodataframe containing filtered entries.
        gpd.GeoDataFrame: Geodataframe containing country polygons of selected continent.
    """

    # get path to file with polygons for which analysis is carried out
    shp_fo = os.path.join(
        root_dir, config["general"]["input_dir"], config["data"]["extent"]["file"]
    )

    # read file
    click.echo(f"Reading extent and spatial aggregation level from file {shp_fo}.")
    extent_gdf = gpd.read_file(shp_fo)

    # fixing invalid geometries
    click.echo("Fixing invalid geometries")
    extent_gdf.geometry = extent_gdf.buffer(0)

    # clip the conflict dataframe to the specified polygons
    click.echo("Clipping conflict dataset to extent.")
    conflict_gdf = gpd.clip(conflict_gdf, extent_gdf)

    return conflict_gdf, extent_gdf
