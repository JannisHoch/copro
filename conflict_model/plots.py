import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sbs
import os, sys
from conflict_model import evaluation

def plot_active_polys(conflict_gdf, extent_gdf, polygon_gdf, config, out_dir, **kwargs):
    """Creates a (1,2)-subplot showing a) selected conflicts and all polygons, and b) selected conflicts and selected polygons.

    Args:
        conflict_gdf (geo-dataframe): geo-dataframe containing the selected conflicts.
        extent_gdf (geo-dataframe): geo-dataframe containing all polygons.
        polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        out_dir (str): path to output folder
    """    

    fig, (ax1, ax2) = plt.subplots(1, 2, **kwargs)
    fig.suptitle('conflict distribution; # conflicts {}; threshold casualties {}; type of violence {}'.format(len(conflict_gdf), config.get('conflict', 'min_nr_casualties'), config.get('conflict', 'type_of_violence')))

    conflict_gdf.plot(ax=ax1, c='r', column='best', cmap='magma', vmin=int(config.get('conflict', 'min_nr_casualties')), vmax=conflict_gdf.best.mean(), legend=True, legend_kwds={'label': "# casualties",})
    extent_gdf.boundary.plot(ax=ax1)
    ax1.set_title('with all polygons')

    conflict_gdf.plot(ax=ax2, c='r', column='best', cmap='magma', vmin=int(config.get('conflict', 'min_nr_casualties')), vmax=conflict_gdf.best.mean(), legend=True, legend_kwds={'label': "# casualties",})
    polygon_gdf.boundary.plot(ax=ax2)
    ax2.set_title('with active polygons only')
                            
    plt.savefig(os.path.join(out_dir, 'conflict_and_casualties_distribution.png'), dpi=300)

    return

def plot_metrics_distribution(out_dict, out_dir, **kwargs):
    """Plots the value distribution of a range of evaluation metrics based on all model simulations.

    Args:
        out_dict (dict): dictionary containing metrics score for various metrics and all simulation.
        out_dir (str): path to output folder.
    """    

    fig, axes = plt.subplots(3, 3, **kwargs)
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

def plot_nr_and_dist_pred(df, gdf, polygon_gdf, out_dir, suffix='', **kwargs):
    """Plots the number of number of predictions made per unique polygon, and the overall value distribution.

    Args:
        df (dataframe): containing model evaluation per unique polygon.
        gdf (geo-dataframe): containing model evaluation per unique polygon.
        polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.
        out_dir (str): path to output folder.
        suffix (str, optional): suffix that can be used to discriminate between different analyses. Defaults to ''.
    """    

    fig, (ax1, ax2) = plt.subplots(1, 2, **kwargs)
    gdf.plot(ax=ax1, column='ID_count', legend=True, cmap='cool')
    polygon_gdf.boundary.plot(ax=ax1, color='0.5')
    ax1.set_title('number of predictions made per polygon')
    sbs.distplot(df.ID_count.values, ax=ax2)
    ax2.set_title('distribution of predictions')
    plt.savefig(os.path.join(out_dir, 'analyis_predictions' + str(suffix) + '.png'), dpi=300)

    return

def plot_frac_and_nr_conf(gdf, polygon_gdf, out_dir, suffix=''):
    """Creates (1,3)-subplot showing per polygon the chance of correct prediction, the number of conflicts, and the chance of correct conflict prediction.

    Args:
        gdf (geo-dataframe): containing model evaluation per unique polygon.
        polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.
        out_dir (str): path to output folder.
        suffix (str, optional): suffix that can be used to discriminate between different analyses. Defaults to ''.
    """    

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    gdf.plot(ax=ax1, column='chance_correct_pred', legend=True, 
                 legend_kwds={'label': "chance correct prediction", 'orientation': "horizontal"})
    polygon_gdf.boundary.plot(ax=ax1, color='0.5')
    gdf.plot(ax=ax2, column='nr_test_confl', legend=True, cmap='Reds', 
                 legend_kwds={'label': "nr of conflicts per polygon", 'orientation': "horizontal"})
    polygon_gdf.boundary.plot(ax=ax2, color='0.5')
    gdf.plot(ax=ax3, column='chance_pred_confl', legend=True, cmap='Blues', 
                 legend_kwds={'label': "chance correct conflict prediction", 'orientation': "horizontal"})
    polygon_gdf.boundary.plot(ax=ax3, color='0.5')
    plt.savefig(os.path.join(out_dir, 'output_evaluation_{}.png'.format(suffix)), dpi=300)

    return

def plot_frac_pred(gdf, gdf_confl, out_dir):
    """Plots the distrubtion of correct predictions for all polygons and only those polygons where conflict was actually observed.

    Args:
        gdf (geo-dataframe): containing model evaluation per unique polygon.
        gdf_confl (geo-dataframe): containing model evaluation per unique polygon where conflict was actually observed.
        out_dir (str): path to output folder
    """    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    sbs.distplot(gdf.chance_correct_pred, ax=ax1)
    sbs.distplot(gdf_confl.chance_correct_pred, ax=ax2)
    plt.savefig(os.path.join(out_dir, 'distribution_chance_correct_pred.png'), dpi=300)

    return

def plot_scatterdata(df, out_dir):
    """Scatterplot of 'ID_count' (number of predictions made per polygon), 'nr_test_confl' (number of conflicts in test-sample), and 'chance_correct_pred'
    (fraction of correct predictions made).

    Args:
        df (dataframe): dataframe with data per polygon.
        out_dir (str): path to output folder
    """    

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    sbs.scatterplot(data=df, x='ID_count', y='chance_correct_pred', ax=ax1)
    sbs.scatterplot(data=df, x='ID_count', y='nr_test_confl', ax=ax2)
    sbs.scatterplot(data=df, x='nr_test_confl', y='chance_correct_pred', ax=ax3)
    plt.savefig(os.path.join(out_dir, 'scatterplot_analysis_all_data.png'), dpi=300)

def plot_correlation_matrix(df, out_dir):
    """Plots the correlation matrix of a dataframe.

    Args:
        df (dataframe): dataframe containing columns to be correlated.
        out_dir (str): path to output folder
    """    

    df_corr = evaluation.correlation_matrix(df)

    fig, (ax1) = plt.subplots(1, 1, figsize=(20, 10))
    sbs.heatmap(df_corr, cmap='YlGnBu', annot=True, cbar=False, ax=ax1)
    plt.savefig(os.path.join(out_dir, 'correlation_matrix.png'), dpi=300)

def plot_categories(gdf, out_dir, mode='median'):
    """Plots the categorization of polygons based on chance of correct prediction and number of conflicts.

    Main categories are:
        * H: chance of correct prediction higher than treshold;
        * L: chance of correct prediction lower than treshold.

    Sub-categories are:
        * HH: high chance of correct prediction with high number of conflicts;
        * HL: high chance of correct prediction with low number of conflicts;
        * LH: low chance of correct prediction with high number of conflicts;
        * LL: low chance of correct prediction with low number of conflicts.

    Args:
        gdf (geo-dataframe): containing model evaluation per unique polygon.
        out_dir (str): path to output folder
        mode (str, optional): Statistical mode used to determine categorization threshold. Defaults to 'median'.
    """    

    gdf = evaluation.categorize_polys(gdf, mode)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    gdf.plot(column='main_category', categorical=True, legend=True, ax=ax1, cmap='copper')
    gdf.plot(column='sub_category', categorical=True, legend=True, ax=ax2, cmap='copper')
    plt.savefig(os.path.join(out_dir, 'polygon_categorization.png'), dpi=300)
    
