import click
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import os

@click.command()
@click.option('-f', '--file_object',help='path to geojson file_object')
@click.option('-c', '--column', help='column name')
@click.option('-t', '--title', help='title for plot and file_object name')
@click.option('-v0', '--minimum-value')
@click.option('-v1', '--maximum-value')
@click.option('-o', '--output-dir', help='path to output directory', type=click.Path())

def main(file_object=None, column=None, title=None, minimum_value=None, maximum_value=None, output_dir=None):
    """Quick and dirty function to plot the column values of a geojson file with minimum user input, and save plot.
    """

    fo = os.path.abspath(file_object)
    click.echo('\nreading csv-file_object {}'.format(fo))
    click.echo('... and converting it to geopandas dataframe')
    df = gpd.read_file(fo)

    click.echo('plotting column {}'.format(column))
    fig, ax = plt.subplots(1, 1)
    df.plot(column=column, 
            ax=ax, 
            cmap='Reds', 
            vmin=minimum_value, vmax=maximum_value,
            legend=True,
            legend_kwds={'label': str(column),
                         'orientation': "vertical"})
    if title != None:
        ax.set_title(str(title))

    file_object_name = str(os.path.basename(file_object)).rsplit('.')[0]
    click.echo('saving plot to file_object {}'.format(os.path.abspath(os.path.join(output_dir, '{}.png'.format(file_object_name)))))
    plt.savefig(os.path.abspath(os.path.join(output_dir, '{}.png'.format(file_object_name))), dpi=300, bbox_inches='tight')

if __name__ == '__main__':

    main()