import click
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import os

@click.command()
@click.argument('file-object', type=click.Path())
@click.option('-c', '--column', help='column name')
@click.option('-t', '--title', help='title for plot and file_object name')
@click.option('-v0', '--minimum-value', default=0, type=float)
@click.option('-v1', '--maximum-value', default=1, type=float)
@click.option('-cmap', '--color-map', default='brg', type=str)
@click.argument('output-dir', type=click.Path())

def main(file_object=None, column=None, title=None, minimum_value=None, maximum_value=None, color_map=None, output_dir=None):
    """Quick and dirty function to plot the column values of a geojson file with minimum user input, and save plot.
    Mainly used for quick inspection of model output in specific years.

    Args:
        file-object (str): path to geoJSON-file whose values are to be plotted.
        output-dir (str): path to directory where plot will be saved.

    Output:
        a png-file of values per polygon.
    """

    # get absolute path to geoJSON-file
    fo = os.path.abspath(file_object)
    click.echo('\nreading file_object {}'.format(fo))
    # read file
    df = gpd.read_file(fo)

    # plot
    click.echo('plotting column {}'.format(column))
    fig, ax = plt.subplots(1, 1)
    df.plot(column=column, 
            ax=ax, 
            cmap=color_map, 
            vmin=minimum_value, vmax=maximum_value,
            legend=True,
            legend_kwds={'label': str(column),
                         'orientation': "vertical"})
    if title != None:
        ax.set_title(str(title))

    # save plot to file
    file_object_name = str(os.path.basename(file_object)).rsplit('.')[0]
    click.echo('saving plot to file_object {}'.format(os.path.abspath(os.path.join(output_dir, '{}.png'.format(file_object_name)))))
    plt.savefig(os.path.abspath(os.path.join(output_dir, '{}.png'.format(file_object_name))), dpi=300, bbox_inches='tight')

if __name__ == '__main__':

    main()