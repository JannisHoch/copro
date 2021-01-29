import click
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import os

@click.command()
@click.option('-f', '--csv-file',help='path to csv file')
@click.option('-c', '--column', help='column name')
@click.option('-t', '--title', help='title for plot and file name')
@click.option('-v0', '--minimum-value')
@click.option('-v1', '--maximum-value')
@click.option('-o', '--output-dir', help='path to output directory', type=click.Path())

def main(csv_file=None, column=None, title=None, minimum_value=None, maximum_value=None, output_dir=None):
    """Quick and dirty function to plot the column values of a shape file with minimum user input, and save plot.
    """

    fo = os.path.abspath(csv_file)
    click.echo('\nreading csv-file {}'.format(fo))
    click.echo('... and converting to dataframe')
    df = pd.read_csv(fo)
    click.echo('... and converting it to geopandas dataframe')
    print(df.head())
    df = gpd.GeoDataFrame(df, geometry=df.geometry)

    click.echo('plotting column {}'.format(column))
    fig, ax = plt.subplots(1, 1)
    df.plot(column=column, 
            ax=ax, 
            cmap='Reds', 
            vmin=minimum_value, vmax=maximum_value,
            legend=True,
            legend_kwds={'label': str(column),
                         'orientation': "vertical"})
    ax.set_title(str(title))

    file_name = 'plt_{0}_fromFile_{1}_andColumn_{2}.png'.format(title, csv_file.rsplit('/')[-1], column)
    click.echo('saving plot to file {}'.format(os.path.abspath(os.path.join(output_dir, file_name))))
    plt.savefig(os.path.abspath(os.path.join(output_dir, file_name)), dpi=300, bbox_inches='tight')

if __name__ == '__main__':

    main()