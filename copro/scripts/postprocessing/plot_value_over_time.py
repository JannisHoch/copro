import click
import glob
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import warnings
from pathlib import Path


def define_val(data_series: pd.Series, statistics: str) -> float:

    if statistics == "mean":
        vals = data_series.mean()
    elif statistics == "median":
        vals = data_series.median()
    elif statistics == "max":
        vals = data_series.max()
    elif statistics == "min":
        vals = data_series.min()
    elif statistics == "std":
        vals = data_series.std()
    elif statistics == "q05":
        vals = data_series.quantile(0.05)
    elif statistics == "q10":
        vals = data_series.quantile(0.1)
    elif statistics == "q90":
        vals = data_series.quantile(0.9)
    elif statistics == "q95":
        vals = data_series.quantile(0.95)
    else:
        raise ValueError(f"{statistics} is not a supported statistical method.")

    return vals


@click.command()
@click.option("-id", "--polygon-id", multiple=True, type=str)
@click.option(
    "-s",
    "--statistics",
    help='which statistical method to use (mean, max, min, std). note: has only effect if with "-id all"!',
    default="mean",
    type=str,
)
@click.option(
    "-c", "--column", help="column name", default="chance_of_conflict", type=str
)
# @click.option("-t", "--title", help="title for plot and file_object name", type=str)
# @click.option("--verbose/--no-verbose", help="verbose on/off", default=False)
@click.argument("input-dir", type=click.Path())
@click.argument("output-dir", type=click.Path())
def main(
    input_dir=None,
    statistics=None,
    polygon_id=None,
    column=None,
    # title=None,
    output_dir=None,
    # verbose=None,
):
    """Quick and dirty function to plot the develoment of a column in the outputted geojson-files over time.
    The script uses all geoJSON-files located in input-dir and retrieves values from them.
    Possible to plot obtain development for multiple polygons (indicated via their ID) or entire study area.
    If the latter, then different statistics can be chosen (mean, max, min, std, median, 'q05', 'q10', 'q90', 'q95').

    Args:
        input-dir (str): path to input directory with geoJSON-files located per projection year.
        output-dir (str): path to directory where output will be stored.

    Output:
        a csv-file containing values per time step.
        a png-file showing development over time.
    """

    click.echo("\nPLOTTING VARIABLE DEVELOPMENT OVER TIME")

    # converting polygon IDs to list
    polygon_id = list(polygon_id)

    # if 'all' is specified, no need to have list but get value directly
    if polygon_id[0] == "all":
        click.echo("INFO: selected entire study area")
        polygon_id = "all"
        click.echo("Selected statistcal method is {}".format(statistics))
        # check if supported statistical function is selected
        if statistics not in [
            "mean",
            "max",
            "min",
            "std",
            "median",
            "q05",
            "q10",
            "q90",
            "q95",
        ]:
            raise ValueError(
                "ERROR: {} is not a supported statistical method".format(statistics)
            )

    # absolute path to input_dir
    input_dir = os.path.abspath(input_dir)
    click.echo(f"Getting geojson-files from {input_dir}.")

    # collect all files in input_dir
    all_files = glob.glob(os.path.join(input_dir, "*.geojson"))

    # create dictionary with list for areas (either IDs or entire study area) to be sampled from
    out_dict = {}
    if polygon_id != "all":
        for idx in polygon_id:
            out_dict[int(idx)] = []
    else:
        out_dict[polygon_id] = []

    # create a list to keep track of year-values in files
    years = []

    # go through all files
    click.echo("INFO: retrieving values from column {}".format(column))
    for geojson in all_files:

        click.echo(f"Reading file {geojson}.")
        # read file and convert to geo-dataframe
        gdf = gpd.read_file(geojson, driver="GeoJSON")
        # convert geo-dataframe to dataframe
        df = pd.DataFrame(gdf.drop(columns="geometry"))

        # get year-value
        year = int(
            str(str(os.path.basename(geojson)).rsplit(".", maxsplit=1)[0]).rsplit(
                "_", maxsplit=1
            )[-1]
        )
        years.append(year)

        # go throough all IDs
        if polygon_id != "all":
            for idx in polygon_id:
                click.echo(f"Sampling ID {idx}.")
                # if ID not in file, assign NaN
                if int(idx) not in df.ID.values:
                    warnings.warn(
                        "ID {} is not in {} - NaN set".format(int(idx), geojson)
                    )
                    vals = np.nan
                # otherwise, get value of column at this ID
                else:
                    vals = df[column].loc[df.ID == int(idx)].values[0]

                # append this value to list in dict
                idx_list = out_dict[int(idx)]
                idx_list.append(vals)

        else:
            # compute mean value over column
            vals = define_val(df[column], statistics)
            # append this value to list in dict
            idx_list = out_dict[polygon_id]
            idx_list.append(vals)

    # create a dataframe from dict and assign year-values as index
    df = pd.DataFrame().from_dict(out_dict)
    years = pd.to_datetime(years, format="%Y")
    df.index = years

    # create an output folder, if not yet there
    Path(os.path.abspath(output_dir)).mkdir(parents=True, exist_ok=True)
    click.echo("Creating output folder {}".format(os.path.abspath(output_dir)))

    # save dataframe as csv-file
    click.echo(
        "Saving to file {}".format(
            os.path.abspath(os.path.join(output_dir, "{}_dev_IDs.csv".format(column)))
        )
    )
    df.to_csv(
        os.path.abspath(os.path.join(output_dir, "{}_dev_IDs.csv".format(column)))
    )

    # create a simple plot and save to file
    # if IDs are specified, with one subplot per ID
    _, axes = plt.subplots(nrows=len(polygon_id), ncols=1, sharex=True)
    df.plot(subplots=True, ax=axes)
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 1))
    plt.savefig(
        os.path.abspath(os.path.join(output_dir, "{}_dev_IDs.png".format(column))),
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":

    main()
