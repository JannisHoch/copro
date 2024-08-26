from copro import conflict, variables, nb, utils
from typing import Tuple, Union
import click
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import os


class XYData:
    def __init__(self, config: dict, target_var: Union[str, None]):
        """Collects feature (X) and target (Y) data for the model.

        Args:
            config (dict): Parsed configuration-settings of the model.
            target_var (Union[str, None]): Target variable of the ML model. Either a string or None. \
                Can be `None` for classification models, but needs to be specified for regression models.
        """
        self.XY_dict = {}
        self.__XY_dict_initiated__ = False
        self.config = config
        self.target_var = target_var

    def _initiate_XY_data(self):

        if self.__XY_dict_initiated__:
            raise ValueError(
                "XY-dictionary already initiated. Please use a new instance of the XY-class."
            )

        # Initialize dictionary
        # some entries are set by default, besides the ones corresponding to input data variables
        self.XY_dict["poly_ID"] = pd.Series()
        self.XY_dict["poly_geometry"] = pd.Series()
        for key in self.config["data"]["indicators"]:
            self.XY_dict[key] = pd.Series(dtype=float)
        self.XY_dict["conflict_t_min_1"] = pd.Series(dtype=bool)
        self.XY_dict["conflict_t_min_1_nb"] = pd.Series(dtype=float)
        # TODO: somewhere a function needs to be added to cater different types of target variables
        # dict key can remain "conflict" but the dtype should be adjusted as it may not be 0/1 anymore
        # could be multi-label classification or regression
        self.XY_dict["conflict"] = pd.Series()

        click.echo("The columns in the sample matrix used are:")
        for key in self.XY_dict:
            click.echo(f"\t{key}.")

        self.__XY_dict_initiated__ = True

    def create_XY(
        self,
        out_dir: click.Path,
        root_dir: click.Path,
        polygon_gdf: gpd.GeoDataFrame,
        conflict_gdf: gpd.GeoDataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Top-level function to create the X-array and Y-array.
        Variable values and conflict data are read from file and stored in array.
        The resulting array is by default saved as npy-format to file.

        Args:
            out_dir (str): path to output folder.
            root_dir (str): path to location of cfg-file.
            polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.
            conflict_gdf (geo-dataframe): geo-dataframe containing the selected conflicts.

        Returns:
            np.ndarray: X-array containing variable values.
            np.ndarray: Y-array containing conflict data.
        """

        self._initiate_XY_data()
        # fill the dictionary and get array
        XY_df = _fill_XY(
            self.XY_dict,
            self.config,
            root_dir,
            conflict_gdf,
            self.target_var,
            polygon_gdf,
            out_dir,
        )

        # save dataframe as geodataframe to GeoPackage in out_dir
        click.echo(
            f"Saving XY data by default to file {os.path.join(out_dir, 'XY.gpkg')}."
        )
        XY_gdf = gpd.GeoDataFrame(XY_df, geometry="poly_geometry")
        XY_gdf.to_file(os.path.join(out_dir, "XY.gpkg"), driver="GPKG")

        # split the XY data into sample data X and target values Y
        X, Y = _split_XY_data(XY_df)

        return X, Y


# def initiate_X_data(config: RawConfigParser) -> dict:
#     """Initiates an empty dictionary to contain the X-data for each polygon, ie. only sample data.
#     This is needed for each time step of each projection run.
#     By default, the first column is for the polygon ID and the second for polygon geometry.
#     The penultimate column is for boolean information about conflict at t-1
#     while the last column is for boolean information about conflict at t-1 in neighboring polygons.
#     All remaining columns correspond to the variables provided in the cfg-file.

#     Args:
#         config (RawConfigParser): object containing the parsed configuration-settings of the model.

#     Returns:
#         dict: emtpy dictionary to be filled, containing keys for each variable (X) plus meta-data.
#     """

#     # Initialize dictionary
#     # some entries are set by default, besides the ones corresponding to input data variables
#     X = {}
#     X["poly_ID"] = pd.Series()
#     X["poly_geometry"] = pd.Series()
#     for key in config.items("data"):
#         X[str(key[0])] = pd.Series(dtype=float)
#     X["conflict_t_min_1"] = pd.Series(dtype=bool)
#     X["conflict_t_min_1_nb"] = pd.Series(dtype=float)

#     click.echo("The columns in the sample matrix used are:")
#     for key in X:
#         click.echo(f"...{key}")

#     return X


def fill_X_sample(
    X: dict,
    config: dict,
    root_dir: str,
    polygon_gdf: gpd.GeoDataFrame,
    proj_year: int,
) -> dict:
    """Fills the X-dictionary with the data sample data besides
    any conflict-related data for each polygon and each year.
    Used during the projection runs as the sample and conflict data
    need to be treated separately there.

    Args:
        X (dict): dictionary containing keys to be sampled.
        config (dict): Parsed configuration-settings of the model.
        root_dir (str): path to location of cfg-file of reference run.
        polygon_gdf (gpd.GeoDataFrame): geo-dataframe containing the selected polygons.
        proj_year (int): year for which projection is made.

    Returns:
        dict: dictionary containing sample values.
    """

    # go through all keys in dictionary
    for key, value in X.items():

        if key == "poly_ID":

            data_series = value
            data_list = utils.get_poly_ID(polygon_gdf)
            data_series = pd.concat(
                [data_series, pd.Series(data_list)], axis=0, ignore_index=True
            )
            X[key] = data_series

        elif key == "poly_geometry":

            data_series = value
            data_list = utils.get_poly_geometry(polygon_gdf)
            data_series = pd.concat(
                [data_series, pd.Series(data_list)], axis=0, ignore_index=True
            )
            X[key] = data_series

        else:

            if key not in ["conflict_t_min_1", "conflict_t_min_1_nb"]:

                nc_ds = xr.open_dataset(
                    os.path.join(
                        root_dir,
                        config["general"]["input_dir"],
                        config["data"]["indicators"][key]["file"],
                    )
                )

                if (np.dtype(nc_ds.time) == np.float32) or (
                    np.dtype(nc_ds.time) == np.float64
                ):
                    data_series = value
                    data_list = variables.nc_with_float_timestamp(
                        polygon_gdf, config, root_dir, key, proj_year
                    )
                    data_series = pd.concat(
                        [data_series, pd.Series(data_list)], axis=0, ignore_index=True
                    )
                    X[key] = data_series

                elif np.dtype(nc_ds.time) == "datetime64[ns]":
                    data_series = value
                    data_list = variables.nc_with_continous_datetime_timestamp(
                        polygon_gdf, config, root_dir, key, proj_year
                    )
                    data_series = pd.concat(
                        [data_series, pd.Series(data_list)], axis=0, ignore_index=True
                    )
                    X[key] = data_series

                else:
                    raise ValueError(
                        "This file has an unsupported dtype for the time variable: {}".format(
                            os.path.join(
                                root_dir,
                                config["general"]["input_dir"],
                                config["data"]["indicators"][key]["file"],
                            )
                        )
                    )

    return X


def fill_X_conflict(
    X: dict, conflict_data: pd.DataFrame, polygon_gdf: gpd.GeoDataFrame
) -> dict:
    """Fills the X-dictionary with the conflict data for each polygon and each year.
    Used during the projection runs as the sample and conflict data need to be treated separately there.

    Args:
        X (dict): dictionary containing keys to be sampled.
        conflict_data (pd.DataFrame): dataframe containing all polygons with conflict.
        polygon_gdf (gpd.GeoDataFrame): geo-dataframe containing the selected polygons.

    Returns:
        dict: dictionary containing sample and conflict values.
    """

    # determine all neighbours for each polygon
    neighboring_matrix = nb.neighboring_polys(polygon_gdf)

    # go through all keys in dictionary
    for key, value in X.items():
        if key == "conflict_t_min_1":

            data_series = value
            data_list = conflict.read_projected_conflict(polygon_gdf, conflict_data)
            data_series = pd.concat(
                [data_series, pd.Series(data_list)], axis=0, ignore_index=True
            )
            X[key] = data_series

        elif key == "conflict_t_min_1_nb":
            data_series = value
            data_list = conflict.read_projected_conflict(
                polygon_gdf,
                conflict_data,
                check_neighbors=True,
                neighboring_matrix=neighboring_matrix,
            )
            data_series = pd.concat(
                [data_series, pd.Series(data_list)], axis=0, ignore_index=True
            )
            X[key] = data_series

        else:
            pass

    return X


def _fill_XY(  # noqa: R0912
    XY: dict,
    config: dict,
    root_dir: click.Path,
    conflict_data: gpd.GeoDataFrame,
    target_var: Union[str, None],
    polygon_gdf: gpd.GeoDataFrame,
    out_dir: click.Path,
) -> pd.DataFrame:
    """Fills the (XY-)dictionary with data for each variable and conflict for each polygon for each simulation year.
    The number of rows should therefore equal to number simulation years times number of polygons.
    At end of last simulation year, the dictionary is converted to a numpy-array.

    Args:
        XY (dict): initiated, i.e. empty, XY-dictionary
        config (dict): Parsed configuration-settings of the model.
        root_dir (str): Path to location of yaml-file.
        conflict_data (gpd.GeoDataFrame): Geodataframe containing the selected conflicts.
        target_var (str): Target variable of the ML model. Either a string or None. \
            Depending on target_var, the conflict data is read differently.
        polygon_gdf (gpd.GeoDataFrame): Geodataframe containing the selected polygons.
        out_dir (path): Path to output folder.

    Returns:
        pd.DataFrame: Dataframe containing the variable values (X) and binary conflict data (Y) plus meta-data.
    """

    # go through all simulation years as specified in config-file
    model_period = np.arange(
        config["general"]["y_start"], config["general"]["y_end"] + 1, 1
    )
    click.echo(f"Reading data for period from {model_period[0]} to {model_period[-1]}.")

    neighboring_matrix = nb.neighboring_polys(polygon_gdf)

    for (sim_year, i) in zip(model_period, range(len(model_period))):

        if i == 0:
            click.echo(f"Skipping first year {sim_year} to start up model.")
        else:
            click.echo(f"Entering year {sim_year}.")
            # go through all keys in dictionary
            for key, value in XY.items():

                if key == "conflict":

                    data_series = value
                    # TODO: guess for target_vars others than None, a dedicasted function is needed
                    if target_var is None:
                        data_list = conflict.conflict_in_year_bool(
                            config, conflict_data, polygon_gdf, sim_year, out_dir
                        )
                    else:
                        raise NotImplementedError(
                            "Implementation of target_var did not happen yet."
                        )
                    data_list = conflict.conflict_in_year_bool(
                        config, conflict_data, polygon_gdf, sim_year, out_dir
                    )
                    data_series = pd.concat(
                        [data_series, pd.Series(data_list)], axis=0, ignore_index=True
                    )
                    XY[key] = data_series

                elif key == "conflict_t_min_1":

                    data_series = value
                    data_list = conflict.conflict_in_previous_year_bool(
                        conflict_data, polygon_gdf, sim_year
                    )
                    data_series = pd.concat(
                        [data_series, pd.Series(data_list)], axis=0, ignore_index=True
                    )
                    XY[key] = data_series

                elif key == "conflict_t_min_1_nb":

                    data_series = value
                    data_list = conflict.conflict_in_previous_year_bool(
                        conflict_data,
                        polygon_gdf,
                        sim_year,
                        check_neighbors=True,
                        neighboring_matrix=neighboring_matrix,
                    )
                    data_series = pd.concat(
                        [data_series, pd.Series(data_list)], axis=0, ignore_index=True
                    )
                    XY[key] = data_series

                elif key == "poly_ID":

                    data_series = value
                    data_list = utils.get_poly_ID(polygon_gdf)
                    data_series = pd.concat(
                        [data_series, pd.Series(data_list)], axis=0, ignore_index=True
                    )
                    XY[key] = data_series

                elif key == "poly_geometry":

                    data_series = value
                    data_list = utils.get_poly_geometry(polygon_gdf)
                    data_series = pd.concat(
                        [data_series, pd.Series(data_list)], axis=0, ignore_index=True
                    )
                    XY[key] = data_series

                else:

                    XY[key] = _read_data_from_netCDF(
                        root_dir, config, key, value, polygon_gdf, sim_year
                    )

            click.echo("All data read.")

    return pd.DataFrame.from_dict(XY)  # .to_numpy()


def _read_data_from_netCDF(
    root_dir: str,
    config: dict,
    key: str,
    value: pd.Series,
    polygon_gdf: gpd.GeoDataFrame,
    sim_year: int,
) -> pd.Series:
    """Reads data from netCDF-file and appends it to the series of the XY-dictionary.
    This happens per variable and simulation year.
    Appends the extracted data to the series of the XY-dictionary.

    .. todo::
        Is the check for different time-dtypes necessary?

    Args:
        root_dir (str): Path to location of yaml-file.
        config (dict):  Parsed configuration-settings of the model.
        key (str): Variable name of feature for which data to be extracted.
        value (pd.Series): Extracted feature values from previous years.
        polygon_gdf (gpd.GeoDataFrame): Geodataframe containing the selected polygons.
        sim_year (int): Simulation year.

    Returns:
        pd.Series: Appended series containing the extracted feature values up to the current simulation year.
    """

    nc_fo = os.path.join(
        root_dir,
        config["general"]["input_dir"],
        config["data"]["indicators"][key]["file"],
    )
    click.echo(f"Reading data for indicator {key} from {nc_fo}.")
    nc_ds = xr.open_dataset(nc_fo)

    if (np.dtype(nc_ds.time) == np.float32) or (np.dtype(nc_ds.time) == np.float64):
        data_series = value
        data_list = variables.nc_with_float_timestamp(
            polygon_gdf, config, root_dir, key, sim_year
        )
        data_series = pd.concat(
            [data_series, pd.Series(data_list)],
            axis=0,
            ignore_index=True,
        )
    elif np.dtype(nc_ds.time) == "datetime64[ns]":
        data_series = value
        data_list = variables.nc_with_continous_datetime_timestamp(
            polygon_gdf, config, root_dir, key, sim_year
        )
        data_series = pd.concat(
            [data_series, pd.Series(data_list)],
            axis=0,
            ignore_index=True,
        )
    else:
        raise ValueError(
            "This file has an unsupported dtype for the time variable: {}".format(
                os.path.join(
                    root_dir,
                    config.get("general", "input_dir"),
                    config.get("data", key),
                )
            )
        )

    return data_series


def _split_XY_data(XY_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separates the XY-array into array containing information about
    variable values (X-array or sample data) and conflict data (Y-array or target data).
    Thereby, the X-array also contains the information about
    unique identifier and polygon geometry.

    Args:
        XY_df (pd.DataFrame): array containing variable values and conflict data.

    Returns:
        pd.DataFrame: X-array, i.e. array containing feature values.
        pd.DataFrame: Y-array, i.e. array containing target values.
    """

    # drop missing values
    XY_df_noNaNs = XY_df.dropna()
    click.echo(
        f"Dropped missing values, which leaves {100 * len(XY_df_noNaNs) / (len(XY_df))} percent of the polygons."
    )

    # get X data
    # since conflict is the last column, we know that all previous columns must be variable values
    X_df = XY_df_noNaNs.iloc[:, :-1]
    # get Y data and convert to integer values
    Y_df = XY_df_noNaNs.iloc[:, -1]
    Y_df = Y_df.astype(int)

    fraction_Y_1 = 100 * len(Y_df[Y_df == 1]) / len(Y_df)
    click.echo(
        f"{round(fraction_Y_1, 2)} percent in the data corresponds to conflicts."
    )

    return X_df, Y_df
