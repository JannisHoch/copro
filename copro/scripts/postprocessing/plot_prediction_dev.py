import click
import glob
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
import os

@click.command()
@click.option('-id', '--polygon-id', multiple=True, type=int)
@click.option('-c', '--column', help='column name', default='chance_of_conflict', type=str)
@click.option('-t', '--title', help='title for plot and file_object name', type=str)
@click.argument('input-dir', type=click.Path())
@click.argument('output-dir', type=click.Path())

def main(input_dir=None, polygon_id=None, column=None, title=None, output_dir=None):
    """Quick and dirty function to plot the develoment of a column in the outputted geojson-files over time.
    """

    assert(len(polygon_id) > 0), AssertionError('please specify one polygon ID to be sampled')

    input_dir = os.path.abspath(input_dir)
    click.echo('\ngetting geojson-files from {}'.format(input_dir))

    all_files = glob.glob(os.path.join(input_dir, '*.geojson'))

    out_dict = dict()
    for idx in polygon_id:
        out_dict[idx] = list()

    years = list()
    
    print('retrieving values from column {}'.format(column))
    for geojson in all_files:
        print('reading file {}'.format(geojson))
        gdf = gpd.read_file(geojson, driver='GeoJSON')
        df = pd.DataFrame(gdf.drop(columns='geometry'))

        year = int(str(str(os.path.basename(geojson)).rsplit('.')[0]).rsplit('_')[-1])
        years.append(year)

        for idx in polygon_id:
            print('sampling ID {}'.format(idx))

            if idx not in df.ID.values: 
                print('WARNING: ID {} is not in {} - NaN set'.format(idx, geojson))
                vals = np.nan
            else:
                vals = df[column].loc[df.ID==idx].values[0]

            idx_list = out_dict[idx]
            idx_list.append(vals)

    df = pd.DataFrame().from_dict(out_dict)
    years = pd.to_datetime(years, format='%Y')
    df.index = years

    fig, axes = plt.subplots(nrows=len(polygon_id), ncols=1, sharex=True)
    df.plot(subplots=True, ax=axes)
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 1))
    if title != None:
        ax.set_title(str(title))
    plt.savefig(os.path.abspath(os.path.join(output_dir, 'prediction_dev.png')), dpi=300, bbox_inches='tight')

if __name__ == '__main__':

    main()