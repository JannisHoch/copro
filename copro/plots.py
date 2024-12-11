import matplotlib.pyplot as plt
import seaborn as sns
from copro import evaluation

sns.set_palette("colorblind")


def selected_polygons(polygon_gdf, **kwargs):
    """Creates a plotting instance of the boundaries of all selected polygons.

    Args:
        polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.

    Kwargs:
        Geopandas-supported keyword arguments.

    Returns:
        ax: Matplotlib axis object.
    """

    ax = polygon_gdf.boundary.plot(**kwargs)

    return ax


def selected_conflicts(conflict_gdf, **kwargs):
    """Creates a plotting instance of the best casualties estimates of the selected conflicts.

    Args:
        conflict_gdf (geo-dataframe): geo-dataframe containing the selected conflicts.

    Kwargs:
        Geopandas-supported keyword arguments.

    Returns:
        ax: Matplotlib axis object.
    """

    ax = conflict_gdf.plot(column="best", **kwargs)

    return ax


def metrics_distribution(out_dict, metrics, **kwargs):
    """Plots the value distribution of a range of evaluation metrics based on all model simulations.

    Args:
        out_dict (dict): dictionary containing metrics score for various metrics and all simulation.

    Kwargs:
        Matplotlib-supported keyword arguments.

    Returns:
        ax: Matplotlib axis object.
    """

    _, ax = plt.subplots(1, 1, **kwargs)

    for metric, color in zip(metrics, sns.color_palette("colorblind")):

        sns.histplot(
            out_dict[str(metric)],
            ax=ax,
            kde=True,
            stat="density",
            color=color,
            label=str(metric),
        )

    plt.legend()

    return ax


def correlation_matrix(df, **kwargs):
    """Plots the correlation matrix of a dataframe.

    Args:
        df (dataframe): dataframe containing columns to be correlated.

    Kwargs:
        Seaborn-supported keyword arguments.

    Returns:
        ax: Matplotlib axis object.
    """

    df_corr = evaluation.calc_correlation_matrix(df)

    ax = sns.heatmap(df_corr, **kwargs)

    return ax


def plot_ROC_curve_n_times(ax, clf, X_test, y_test, tprs, aucs, mean_fpr, **kwargs):
    """Plots the ROC-curve per model simulation to a pre-initiated matplotlib-instance.

    Args:
        ax (axis): axis of pre-initaited matplotlib-instance
        clf (classifier): sklearn-classifier used in the simulation.
        X_test (array): array containing test-sample variable values.
        y_test (list): list containing test-sample conflict data.
        tprs (list): list with false positive rates.
        aucs (list): list with area-under-curve values.
        mean_fpr (array): array with mean false positive rate.

    Returns:
        list: lists with true positive rates and area-under-curve values per plot.
    """

    raise NotImplementedError(
        "Plotting API in sklearn is changed, function needs updating and reimplementation."
    )


def plot_ROC_curve_n_mean(ax, tprs, aucs, mean_fpr, **kwargs):
    """Plots the mean ROC-curve to a pre-initiated matplotlib-instance.

    Args:
        ax (axis): axis of pre-initaited matplotlib-instance
        tprs (list): list with false positive rates.
        aucs (list): list with area-under-curve values.
        mean_fpr (array): array with mean false positive rate.
    """

    raise NotImplementedError(
        "Plotting API in sklearn is changed, function needs updating and reimplementation."
    )
