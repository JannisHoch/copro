import glob
import os
from shutil import rmtree

import click
import geopandas as gpd
import matplotlib.pyplot as plt
from PIL import Image


@click.command()
@click.option(
    "-c", "--column", help="column name", default="chance_of_conflict", type=str
)
@click.option("-cmap", "--color-map", default="brg", type=str)
@click.option("-v0", "--minimum-value", default=0, type=float)
@click.option("-v1", "--maximum-value", default=1, type=float)
@click.option(
    "--delete/--no-delete", help="whether or not to delete png-files", default=True
)
@click.argument("input-dir", type=click.Path())
@click.argument("output-dir", type=click.Path())
def main(
    input_dir=None,
    column=None,
    color_map=None,
    minimum_value=None,
    maximum_value=None,
    delete=None,
    output_dir=None,
):
    """Function to convert column values of all geoJSON-files in a directory into one GIF-file.
    The function provides several options to modify the design of the GIF-file.
    The GIF-file is based on png-files of column value per geoJSON-file.
    It is possible to keep these png-file as simple plots of values per time step.

    Args:
        input-dir (str): path to directory where geoJSON-files are stored.
        output_dir (str): path to directory where GIF-file will be stored.

    Output:
        GIF-file with animated column values per input geoJSON-file.
    """

    # get path to geoJSON-files
    input_dir = os.path.abspath(input_dir)
    click.echo("\ngetting geojson-files from {}".format(input_dir))

    # create folder where intermediate png-files are stored
    png_dir = os.path.join(output_dir, "png")
    click.echo("creating png-folder {}".format(png_dir))
    if not os.path.isdir(png_dir):
        os.mkdir(png_dir)

    # collect all geoJSON-files
    all_files = glob.glob(os.path.join(input_dir, "*.geojson"))

    # plot values per geoJSON-file
    click.echo("plotting column {}".format(column))
    for geojson in all_files:
        click.echo("reading file {}".format(geojson))
        # read geoJSON-file
        gdf = gpd.read_file(geojson, driver="GeoJSON")
        # retrieve year information from filename of geoJSON-file
        year = int(
            str(str(os.path.basename(geojson)).rsplit(".", maxsplit=1)[0]).rsplit(
                "_", maxsplit=1
            )[-1]
        )

        _, ax = plt.subplots(1, 1)
        gdf.plot(
            column=column,
            ax=ax,
            cmap=color_map,
            vmin=minimum_value,
            vmax=maximum_value,
            legend=True,
            legend_kwds={"label": str(column), "orientation": "vertical"},
        )

        ax.set_title(str(year))
        # save plot
        click.echo("saving plot to png-folder")
        plt.savefig(
            os.path.join(png_dir, "plt_{}_{}.png".format(column, year)),
            dpi=300,
            bbox_inches="tight",
        )

    # create GIF and save
    click.echo("creating GIF from saved plots")
    # based on: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    fp_in = os.path.join(png_dir, "*_{}_*.png".format(column))
    fp_out = os.path.join(output_dir, "{}_over_time.gif".format(column))
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(
        fp=fp_out, format="GIF", append_images=imgs, save_all=True, duration=500, loop=0
    )

    # if specified, delete all (intermediate) png-files
    if delete:
        click.echo("removing png-folder")
        rmtree(png_dir)


if __name__ == "__main__":

    main()
