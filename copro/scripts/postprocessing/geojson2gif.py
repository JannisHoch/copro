import click
import glob
import matplotlib.pyplot as plt
import geopandas as gpd
from shutil import rmtree
from PIL import Image
import os

@click.command()
@click.option('-c', '--column', help='column name', default='chance_of_conflict', type=str)
@click.option('-cmap', '--color-map', default='brg', type=str)
@click.option('-v0', '--minimum-value', default=0, type=float)
@click.option('-v1', '--maximum-value', default=1, type=float)
@click.option('--delete/--no-delete', help='whether or not to delete png-files', default=True)
@click.argument('input-dir', type=click.Path())
@click.argument('output-dir', type=click.Path())

def main(input_dir=None, column=None, color_map=None, minimum_value=None, maximum_value=None, delete=None, output_dir=None):
    """Quick and dirty function to convert all geojson files into one GIF animation.
    """

    input_dir = os.path.abspath(input_dir)
    click.echo('\ngetting geojson-files from {}'.format(input_dir))

    png_dir = os.path.join(output_dir, 'png')
    click.echo('creating png-folder {}'.format(png_dir))
    if not os.path.isdir(png_dir):
        os.mkdir(png_dir)

    all_files = glob.glob(os.path.join(input_dir, '*.geojson'))
    
    click.echo('plotting column {}'.format(column))
    for geojson in all_files:
        click.echo('reading file {}'.format(geojson))
        gdf = gpd.read_file(geojson, driver='GeoJSON')

        year = int(str(str(os.path.basename(geojson)).rsplit('.')[0]).rsplit('_')[-1])

        fig, ax = plt.subplots(1, 1)
        gdf.plot(column=column, 
                 ax=ax, 
                 cmap=color_map, 
                 vmin=minimum_value, vmax=maximum_value,
                 legend=True,
                 legend_kwds={'label': str(column),
                              'orientation': "vertical"})

        ax.set_title(str(year))
        click.echo('saving plot to png-folder')
        plt.savefig(os.path.join(png_dir, 'plt_{}_{}.png'.format(column, year)), dpi=300, bbox_inches='tight')
    
    click.echo('creating GIF from saved plots')
    # based on: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    fp_in = os.path.join(png_dir, '*_{}_*.png'.format(column))
    fp_out = os.path.join(output_dir, '{}_over_time.gif'.format(column))
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=500, loop=0)

    if delete:
        click.echo('removing png-folder')
        rmtree(png_dir)

if __name__ == '__main__':

    main()