import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sbs
import os, sys

def plot_active_polys(conflict_gdf, extent_gdf, extent_active_polys_gdf, config, out_dir):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    fig.suptitle('conflict distribution; # conflicts {}; threshold casualties {}; type of violence {}'.format(len(conflict_gdf), config.get('conflict', 'min_nr_casualties'), config.get('conflict', 'type_of_violence')))

    conflict_gdf.plot(ax=ax1, c='r', column='best', cmap='magma', vmin=int(config.get('conflict', 'min_nr_casualties')), vmax=conflict_gdf.best.mean(), legend=True, legend_kwds={'label': "# casualties",})
    extent_gdf.boundary.plot(ax=ax1)
    ax1.set_title('with all polygons')

    conflict_gdf.plot(ax=ax2, c='r', column='best', cmap='magma', vmin=int(config.get('conflict', 'min_nr_casualties')), vmax=conflict_gdf.best.mean(), legend=True, legend_kwds={'label': "# casualties",})
    extent_active_polys_gdf.boundary.plot(ax=ax2)
    ax2.set_title('with active polygons only')
                            
    plt.savefig(os.path.join(out_dir, 'conflict_and_casualties_distribution.png'), dpi=300)

    return

def plot_metrics_distribution(out_dict, out_dir):

    fig, axes = plt.subplots(3, 3, figsize=(20, 10))
    sbs.distplot(out_dict['Accuracy'], ax=axes[0,0], color="k")
    axes[0,0].set_title('Accuracy')
    sbs.distplot(out_dict['Precision'], ax=axes[0,1], color="r")
    axes[0,1].set_title('Precision')
    sbs.distplot(out_dict['Recall'], ax=axes[0,2], color="b")
    axes[0,2].set_title('Recall')
    sbs.distplot(out_dict['F1 score'], ax=axes[1,0], color="g")
    axes[1,0].set_title('F1 score')
    sbs.distplot(out_dict['Cohen-Kappa score'], ax=axes[1,1], color="c")
    axes[1,1].set_title('Cohen-Kappa score')
    sbs.distplot(out_dict['Brier loss score'], ax=axes[1,2], color="y")
    axes[1,2].set_title('Brier loss score')
    sbs.distplot(out_dict['ROC AUC score'], ax=axes[2,0], color="k")
    axes[2,0].set_title('ROC AUC score')
    plt.savefig(os.path.join(out_dir, 'distribution_output_evaluation_criteria.png'), dpi=300)

    return

def plot_nr_and_dist_pred(df_hit, gdf_hit, extent_active_polys_gdf, out_dir, suffix=''):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    gdf_hit.plot(ax=ax1, column='ID_count', legend=True, cmap='cool')
    extent_active_polys_gdf.boundary.plot(ax=ax1, color='0.5')
    ax1.set_title('number of predictions made per polygon')
    sbs.distplot(df_hit.ID_count.values, ax=ax2)
    ax2.set_title('distribution of predictions')
    plt.savefig(os.path.join(out_dir, 'analyis_predictions' + str(suffix) + '.png'), dpi=300)

    return

def plot_frac_and_nr_conf(gdf_hit, extent_active_polys_gdf, out_dir, suffix=''):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    gdf_hit.plot(ax=ax1, column='average_hit', legend=True, figsize=(20,10))
    extent_active_polys_gdf.boundary.plot(ax=ax1, color='0.5')
    ax1.set_title('fraction of all correct predictions made')
    gdf_hit.plot(ax=ax2, column='nr_of_test_confl', legend=True, cmap='Reds')
    extent_active_polys_gdf.boundary.plot(ax=ax2, color='0.5')
    ax2.set_title('nr of conflicts per polygon')
    plt.savefig(os.path.join(out_dir, 'average_hit_precision' + str(suffix) + '.png'), dpi=300)

    return

def plot_frac_pred(gdf_hit, gdf_hit_1, out_dir):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    sbs.distplot(gdf_hit.average_hit, ax=ax1)
    sbs.distplot(gdf_hit_1.average_hit, ax=ax2)
    plt.savefig(os.path.join(out_dir, 'distribution_average_hit.png'), dpi=300)

    return

def plot_scatterdata(df_hit, out_dir):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    sbs.scatterplot(data=df_hit, x='ID_count', y='average_hit', ax=ax1)
    sbs.scatterplot(data=df_hit, x='total_hits', y='average_hit', ax=ax2)
    sbs.scatterplot(data=df_hit, x='nr_of_test_confl', y='average_hit', ax=ax3)
    plt.savefig(os.path.join(out_dir, 'scatterplot_analysis_all_data'), dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    sbs.scatterplot(data=df_hit, x='total_hits', y='average_hit', hue='nr_of_test_confl', size='nr_of_test_confl', ax=ax)
    plt.savefig(os.path.join(out_dir, 'scatterplot_total_hits_and_average_hit.png'), dpi=300)
