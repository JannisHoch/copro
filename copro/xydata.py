from copro import conflict, variables, nb, utils
from configparser import RawConfigParser
from typing import Tuple
import click
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import os
import warnings


class XYData:
    def __init__(self, config: RawConfigParser):
        self.XY_dict = {}
        self.__XY_dict_initiated__ = False
        self.config = config

    def _initiate_XY_data(self):

        if self.__XY_dict_initiated__:
            raise ValueError(
                "XY-dictionary already initiated. Please use a new instance of the XY-class."
            )

        # Initialize dictionary
        # some entries are set by default, besides the ones corresponding to input data variables
        self.XY_dict["poly_ID"] = pd.Series()
        self.XY_dict["poly_geometry"] = pd.Series()
        for key in self.config.items("data"):
            self.XY_dict[str(key[0])] = pd.Series(dtype=float)
        self.XY_dict["conflict_t_min_1"] = pd.Series(dtype=bool)
        self.XY_dict["conflict_t_min_1_nb"] = pd.Series(dtype=float)
        self.XY_dict["conflict"] = pd.Series(dtype=bool)

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
    ) -> Tuple[np.array, np.array]:
        """Top-level function to create the X-array and Y-array.
        If the XY-data was pre-computed and specified in cfg-file, the data is loaded.
        If not, variable values and conflict data are read from file and stored in array.
        The resulting array is by default saved as npy-format to file.

        Args:
            config (ConfigParser-object): object containing the parsed configuration-settings of the model.
            out_dir (str): path to output folder.
            root_dir (str): path to location of cfg-file.
            polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.
            conflict_gdf (geo-dataframe): geo-dataframe containing the selected conflicts.

        Returns:
            array: X-array containing variable values.
            array: Y-array containing conflict data.
        """

        # if nothing is specified in cfg-file, then initiate and fill XY data from scratch
        if self.config.get("pre_calc", "XY") != " ":
            self._initiate_XY_data()
            # fill the dictionary and get array
            XY_arr = _fill_XY(
                self.XY_dict, self.config, root_dir, conflict_gdf, polygon_gdf, out_dir
            )
            # save array to XY.npy out_dir
            click.echo(
                f"Saving XY data by default to file {os.path.join(out_dir, 'XY.npy')}."
            )
            np.save(os.path.join(out_dir, "XY"), XY_arr)
        # if path to XY.npy is specified, read the data intead
        else:
            click.echo(
                f"Loading XY data from file {os.path.join(root_dir, self.config.get('pre_calc', 'XY'))}."
            )
            XY_arr = np.load(
                os.path.join(root_dir, self.config.get("pre_calc", "XY")),
                allow_pickle=True,
            )

        # split the XY data into sample data X and target values Y
        X, Y = _split_XY_data(XY_arr)

        return X, Y


def initiate_X_data(config: RawConfigParser) -> dict:
    """Initiates an empty dictionary to contain the X-data for each polygon, ie. only sample data.
    This is needed for each time step of each projection run.
    By default, the first column is for the polygon ID and the second for polygon geometry.
    The penultimate column is for boolean information about conflict at t-1
    while the last column is for boolean information about conflict at t-1 in neighboring polygons.
    All remaining columns correspond to the variables provided in the cfg-file.

    Args:
        config (RawConfigParser): object containing the parsed configuration-settings of the model.

    Returns:
        dict: emtpy dictionary to be filled, containing keys for each variable (X) plus meta-data.
    """

    # Initialize dictionary
    # some entries are set by default, besides the ones corresponding to input data variables
    X = {}
    X["poly_ID"] = pd.Series()
    X["poly_geometry"] = pd.Series()
    for key in config.items("data"):
        X[str(key[0])] = pd.Series(dtype=float)
    X["conflict_t_min_1"] = pd.Series(dtype=bool)
    X["conflict_t_min_1_nb"] = pd.Series(dtype=float)

    if config.getboolean("general", "verbose"):
        click.echo("DEBUG: the columns in the sample matrix used are:")
        for key in X:
            click.echo("...{}".format(key))

    return X


def fill_X_sample(
    X: dict,
    config: RawConfigParser,
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
        config (RawConfigParser): object containing the parsed configuration-settings of the model.
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
                        config.get("general", "input_dir"),
                        config.get("data", key),
                    ).rsplit(",")[0]
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
                    raise Warning(
                        "This file has an unsupported dtype for the time variable: {}".format(
                            os.path.join(
                                root_dir,
                                config.get("general", "input_dir"),
                                config.get("data", key),
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
    config: RawConfigParser,
    root_dir: click.Path,
    conflict_data: gpd.GeoDataFrame,
    polygon_gdf: gpd.GeoDataFrame,
    out_dir: click.Path,
) -> np.ndarray:
    """Fills the (XY-)dictionary with data for each variable and conflict for each polygon for each simulation year.
    The number of rows should therefore equal to number simulation years times number of polygons.
    At end of last simulation year, the dictionary is converted to a numpy-array.

    Args:
        XY (dict): initiated, i.e. empty, XY-dictionary
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        root_dir (str): path to location of cfg-file.
        conflict_data (geo-dataframe): geo-dataframe containing the selected conflicts.
        polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.
        out_dir (path): path to output folder.

    Returns:
        array: filled array containing the variable values (X) and binary conflict data (Y) plus meta-data.
    """

    # go through all simulation years as specified in config-file
    model_period = np.arange(
        config.getint("settings", "y_start"), config.getint("settings", "y_end") + 1, 1
    )
    click.echo(f"Reading data for period from {model_period[0]} to {model_period[-1]}.")

    neighboring_matrix = nb.neighboring_polys(polygon_gdf)

    for (sim_year, i) in zip(model_period, range(len(model_period))):

        if i == 0:
            click.echo(f"Skipping first year {sim_year} to start up model")
        else:
            click.echo(f"Entering year {sim_year}.")
            # go through all keys in dictionary
            for key, value in XY.items():

                if key == "conflict":

                    data_series = value
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

                    nc_ds = xr.open_dataset(
                        os.path.join(
                            root_dir,
                            config.get("general", "input_dir"),
                            config.get("data", key),
                        ).rsplit(",")[0]
                    )

                    if (np.dtype(nc_ds.time) == np.float32) or (
                        np.dtype(nc_ds.time) == np.float64
                    ):
                        data_series = value
                        data_list = variables.nc_with_float_timestamp(
                            polygon_gdf, config, root_dir, key, sim_year
                        )
                        data_series = pd.concat(
                            [data_series, pd.Series(data_list)],
                            axis=0,
                            ignore_index=True,
                        )
                        XY[key] = data_series

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
                        XY[key] = data_series

                    else:
                        warnings.warn(
                            "This file has an unsupported dtype for the time variable: {}".format(
                                os.path.join(
                                    root_dir,
                                    config.get("general", "input_dir"),
                                    config.get("data", key),
                                )
                            )
                        )

            click.echo("All data read.")

    df_out = pd.DataFrame.from_dict(XY)

    return df_out.to_numpy()


def _split_XY_data(XY_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Separates the XY-array into array containing information about
    variable values (X-array or sample data) and conflict data (Y-array or target data).
    Thereby, the X-array also contains the information about
    unique identifier and polygon geometry.

    Args:
        XY (array): array containing variable values and conflict data.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Returns:
        arrays: two separate arrays, the X-array and Y-array.
    """

    # convert array to dataframe for easier handling
    XY_df = pd.DataFrame(XY_arr)
    # fill all missing values with 0
    XY_df = XY_df.fillna(0)
    # convert dataframe back to array
    XY_df = XY_df.to_numpy()

    # get X data
    # since conflict is the last column, we know that all previous columns must be variable values
    X = XY_df[:, :-1]
    # get Y data and convert to integer values
    Y = XY_df[:, -1]
    Y = Y.astype(int)

    fraction_Y_1 = 100 * len(np.where(Y != 0)[0]) / len(Y)
    click.echo(
        f"{round(fraction_Y_1, 2)} percent in the data corresponds to conflicts."
    )

    return X, Y
