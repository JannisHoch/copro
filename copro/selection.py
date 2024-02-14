import geopandas as gpd
import pandas as pd
import os
from copro import utils
from configparser import RawConfigParser
import click
from typing import Tuple
from ast import literal_eval


def select(
    config: RawConfigParser, out_dir: click.Path, root_dir: click.Path
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    """Main function performing the selection procedure.
    First, selects only conflicts matching specified properties.
    Second, clips the conflict data to a specified spatial extent.
    Third, retrieves the geometry of all polygons in the spatial extent and assigns IDs.

    Args:
        config (RawConfigParser): object containing the parsed configuration-settings of the model.
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
    gdf: gpd.GeoDataFrame, config: RawConfigParser
) -> gpd.GeoDataFrame:
    """Filters conflict database according to certain conflict properties
    such as number of casualties, type of violence or country.

    Args:
        gdf (gpd.GeoDataFrame): geo-dataframe containing entries with conflicts.
        config (RawConfigParser): object containing the parsed configuration-settings of the model.

    Returns:
        gpd.GeoDataFrame: geo-dataframe containing filtered entries.
    """

    # create dictionary with all selection criteria
    selection_criteria = {
        "best": config.getint("conflict", "min_nr_casualties"),
        "type_of_violence": (config.get("conflict", "type_of_violence")).rsplit(","),
    }

    click.echo("Filtering based on conflict properties.")
    # go through all criteria
    for key, value in selection_criteria.items():

        # for criterion 'best' (i.e. best estimate of fatalities), select all entries above threshold
        if key == "best" and value != "":
            click.echo(f"Filtering key {key} with lower value {value}.")
            gdf = gdf[gdf["best"] >= value]
        # for other criteria, select all entries matching the specified value(s) per criterion
        if key == "type_of_violence" and value != "":
            click.echo(f"Filtering key {key} with value(s) {value}.")
            # NOTE: check if this works like this
            values = [literal_eval(i) for i in value]
            gdf = gdf[gdf[key].isin(values)]

    return gdf


def _clip_to_extent(
    conflict_gdf: gpd.GeoDataFrame, config: RawConfigParser, root_dir: click.Path
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """As the original conflict data has global extent, this function clips the database
    to those entries which have occured on a specified continent.

    Args:
        conflict_gdf (geo-dataframe): geo-dataframe containing entries with conflicts.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): path to location of cfg-file.

    Returns:
        geo-dataframe: geo-dataframe containing filtered entries.
        geo-dataframe: geo-dataframe containing country polygons of selected continent.
    """

    # get path to file with polygons for which analysis is carried out
    shp_fo = os.path.join(
        root_dir, config.get("general", "input_dir"), config.get("extent", "shp")
    )

    # read file
    click.echo(f"Reading extent and spatial aggregation level from file {shp_fo}.")
    extent_gdf = gpd.read_file(shp_fo)

    # fixing invalid geometries
    click.echo("Fixing invalid geometries")
    extent_gdf.geometry = extent_gdf.buffer(0)

    # clip the conflict dataframe to the specified polygons
    click.echo("Clipping clipping conflict dataset to extent.")
    conflict_gdf = gpd.clip(conflict_gdf, extent_gdf)

    return conflict_gdf, extent_gdf
