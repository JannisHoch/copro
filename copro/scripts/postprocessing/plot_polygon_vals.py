import click
import matplotlib.pyplot as plt
import geopandas as gpd
import os

@click.command()
@click.option('-s', '--shp-file',help='path to shp file')
@click.option('-c', '--column', help='column name')
@click.option('-t', '--title', help='title for plot and file name')
@click.option('-v0', '--minimum-value')
@click.option('-v1', '--maximum-value')
@click.option('-o', '--output-dir', help='path to output directory', type=click.Path())

def main(shp_file=None, column=None, title=None, minimum_value=None, maximum_value=None, output_dir=None):
    """Quick and dirty function to plot the column values of a shape file with minimum user input, and save plot.
    """

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    africa = world[world.continent=='Africa']

    fo = os.path.abspath(shp_file)
    click.echo('\nreading shp-file {}'.format(fo))
    click.echo('... and converting it to geopandas dataframe')
    df = gpd.read_file(fo)

    click.echo('plotting column {}'.format(column))
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    df.plot(column=column, 
            ax=ax, 
            cmap='Reds', 
            vmin=minimum_value, vmax=maximum_value,
            legend=True,
            legend_kwds={'label': str(column),
                         'orientation': "vertical"})
    ax.set_title(str(title), fontsize=20)

    africa.boundary.plot(ax=ax, color='0.5')

    file_name = 'plt_{0}_fromFile_{1}_andColumn_{2}.png'.format(title, shp_file.rsplit('/')[-1], column)
    click.echo('saving plot to file {}'.format(os.path.abspath(os.path.join(output_dir, file_name))))
    plt.savefig(os.path.abspath(os.path.join(output_dir, file_name)), dpi=300, bbox_inches='tight')

if __name__ == '__main__':

    main()