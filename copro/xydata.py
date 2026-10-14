import os
import pickle
from typing import Tuple, Union

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from copro import conflict, nb, utils, variables

# Conflict-derived feature columns (Step 3 added the last four). These are
# filled from conflict data, never from NetCDF, in both reference and
# projection runs.
CONFLICT_DERIVED_KEYS = [
    "conflict_t_min_1",
    "conflict_t_min_1_nb",
    "conflict_decay_hl2",
    "conflict_decay_hl8",
]

# Row loss above this fraction is reported loudly. Any whole polygon lost is
# reported regardless of fraction - a polygon absent from the matrix is absent
# from every downstream product, which is a data decision, not a rounding
# detail.
DROP_WARN_ROW_FRACTION = 0.05
 
# Cap on how many polygon IDs are listed in the message. At global scale the
# list would otherwise run to hundreds of lines.
DROP_REPORT_MAX_IDS = 25

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
        # conflict-history features (Step 3): VIEWS-style time-since-conflict
        # decay at two half-lives. (GED sub-threshold counts were tested and
        # dropped: UCDP's dyad gate biases sub-threshold counts toward dyads
        # that later escalate, and the counts have no forward analogue in
        # multi-year scenario projection.)
        self.XY_dict["conflict_decay_hl2"] = pd.Series(dtype=float)
        self.XY_dict["conflict_decay_hl8"] = pd.Series(dtype=float)
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
        write_xy_gpkg: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Top-level function to create the X-array and Y-array.
        Variable values and conflict data are read from file and stored in array.
        The resulting array is by default saved as npy-format to file.

        Args:
            out_dir (str): path to output folder.
            root_dir (str): path to location of cfg-file.
            polygon_gdf (geo-dataframe): geo-dataframe containing the selected polygons.
            conflict_gdf (geo-dataframe): geo-dataframe containing the selected conflicts.

        Returns:
            pd.DataFrame: dataframe containing feature (X) values.
            pd.DataFrame: dataframe containing conflict (Y) data.
        """

        self._initiate_XY_data()

        # path to a pre-computed XY cache, if any. "" means no cache.
        xy_cache = self.config.get("pre_calc", {}).get("XY", "")

        if xy_cache and os.path.isfile(xy_cache):
            # REUSE: load cached matrix, skip every NetCDF read
            click.echo(f"Loading pre-computed XY from cache {xy_cache}.")
            with open(xy_cache, "rb") as f:
                cached = pickle.load(f)
            _check_XY_cache(cached, self.config)
            XY_df = cached["XY_df"]
            # _fill_XY normally writes the y_end files used as projection
            # initial conditions (conflicts_in_, last_conflict_year_); the
            # cache path skips _fill_XY and the _REF folder is recreated each
            # run, so regenerate them here.
            _write_yend_conflict_from_cache(cached, self.config, out_dir)
        else:
            # COMPUTE: read all indicators from NetCDF as before
            XY_df = _fill_XY(
                self.XY_dict,
                self.config,
                root_dir,
                conflict_gdf,
                polygon_gdf,
                out_dir,
            )
            # if a cache path was given but no file existed yet, write it now
            if xy_cache:
                click.echo(f"Writing XY cache to {xy_cache}.")
                os.makedirs(os.path.dirname(xy_cache), exist_ok=True)
                with open(xy_cache, "wb") as f:
                    pickle.dump(
                        {
                            "XY_df": XY_df,
                            "columns": list(XY_df.columns),
                            "y_start": self.config["general"]["y_start"],
                            "y_end": self.config["general"]["y_end"],
                            "polygon_file": self.config["data"]["extent"]["file"],
                            "conflict_file": self.config["data"]["conflict"]["file"],
                        },
                        f,
                    )

        # save dataframe as geodataframe to GeoPackage in out_dir.
        # The deploy fit switches this off: the file is ~692 MB per region and
        # nothing reads it.
        if write_xy_gpkg:
            click.echo(
                f"Saving XY data by default to file {os.path.join(out_dir, 'XY.gpkg')}."
            )
            XY_gdf = gpd.GeoDataFrame(XY_df, geometry="poly_geometry")
            XY_gdf.to_file(os.path.join(out_dir, "XY.gpkg"), driver="GPKG")
            
        # split the XY data into sample data X and target values Y
        X, Y, meta = _split_XY_data(XY_df)

        return X, Y, meta


def initiate_X_data(config: dict) -> dict:
    """Initiates an empty dictionary to contain the X-data for each polygon, ie. only sample data.
    This is needed for each time step of each projection run.
    By default, the first column is for the polygon ID and the second for polygon geometry.
    The penultimate column is for boolean information about conflict at t-1
    while the last column is for boolean information about conflict at t-1 in neighboring polygons.
    All remaining columns correspond to the variables provided in the cfg-file.

    ..todo::
        Can this be better aligned with the XYData-class?

    Args:
        config (dict): object containing the parsed configuration-settings of the model.

    Returns:
        dict: emtpy dictionary to be filled, containing keys for each variable (X) plus meta-data.
    """

    # Initialize dictionary
    # some entries are set by default, besides the ones corresponding to input data variables
    X = {}
    X["poly_ID"] = pd.Series()
    X["poly_geometry"] = pd.Series()
    for key in config["data"]["indicators"].keys():
        X[key] = pd.Series(dtype=float)
    X["conflict_t_min_1"] = pd.Series(dtype=bool)
    X["conflict_t_min_1_nb"] = pd.Series(dtype=float)
    X["conflict_decay_hl2"] = pd.Series(dtype=float)
    X["conflict_decay_hl8"] = pd.Series(dtype=float)

    click.echo("The columns in the sample matrix used are:")
    for key in X:
        click.echo(f"...{key}")

    return X


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

            if key not in CONFLICT_DERIVED_KEYS:

                X[key] = _read_data_from_netCDF(
                    root_dir, config, key, value, polygon_gdf, proj_year
                )

    return X


def fill_X_conflict(
    X: dict,
    conflict_data: pd.DataFrame,
    polygon_gdf: gpd.GeoDataFrame,
    proj_year: int,
    last_conflict_lookup: dict,
) -> dict:
    """Fills the X-dictionary with the conflict data for each polygon and each year.
    Used during the projection runs as the sample and conflict data need to be treated separately there.

    Args:
        X (dict): dictionary containing keys to be sampled.
        conflict_data (pd.DataFrame): dataframe containing all polygons with conflict.
        polygon_gdf (gpd.GeoDataFrame): geo-dataframe containing the selected polygons.
        proj_year (int): projection year being filled.
        last_conflict_lookup (dict): watprovID -> last conflict year <= proj_year - 1 \
            (observed history plus this estimator's own projected trajectory).

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

        elif key == "conflict_decay_hl2":
            data_list = conflict.projected_decay(
                polygon_gdf, last_conflict_lookup, proj_year, 2.0
            )
            X[key] = pd.concat(
                [value, pd.Series(data_list)], axis=0, ignore_index=True
            )

        elif key == "conflict_decay_hl8":
            data_list = conflict.projected_decay(
                polygon_gdf, last_conflict_lookup, proj_year, 8.0
            )
            X[key] = pd.concat(
                [value, pd.Series(data_list)], axis=0, ignore_index=True
            )

        else:
            pass

    return X

def _check_XY_cache(cached: dict, config: dict):
    """Validates a loaded XY cache against the current configuration.
    Raises loudly on any mismatch so a stale cache is never used silently.
    """
    expected_cols = (
        ["year", "poly_ID", "poly_geometry"]
        + list(config["data"]["indicators"].keys())
        + CONFLICT_DERIVED_KEYS
        + ["conflict"]
    )
    if cached["columns"] != expected_cols:
        raise ValueError(
            "XY cache column mismatch - indicator set or order changed.\n"
            f"  cached:   {cached['columns']}\n"
            f"  expected: {expected_cols}\n"
            "Delete the cache file and re-run to rebuild it."
        )
    if cached["y_start"] != config["general"]["y_start"]:
        raise ValueError(
            f"XY cache y_start mismatch: cached {cached['y_start']}, "
            f"config {config['general']['y_start']}. Delete the cache and re-run."
        )
    if cached["y_end"] != config["general"]["y_end"]:
        raise ValueError(
            f"XY cache y_end mismatch: cached {cached['y_end']}, "
            f"config {config['general']['y_end']}. Delete the cache and re-run."
        )
    if cached["polygon_file"] != config["data"]["extent"]["file"]:
        raise ValueError(
            "XY cache polygon_file mismatch: "
            f"cached {cached['polygon_file']}, "
            f"config {config['data']['extent']['file']}. "
            "Delete the cache and re-run."
        )
    if cached["conflict_file"] != config["data"]["conflict"]["file"]:
        raise ValueError(
            "XY cache conflict_file mismatch: "
            f"cached {cached['conflict_file']}, "
            f"config {config['data']['conflict']['file']}. "
            "Delete the cache and re-run."
        )


def _write_yend_conflict_from_cache(
    cached: dict, config: dict, out_dir: click.Path
):
    """Regenerates the projection initial-condition files from the cached XY
    matrix. Loading the cache skips _fill_XY, which is what normally writes
    them. Files (formats match the _fill_XY originals):
    - conflicts_in_<y_end>.csv: polygons with conflict in y_end.
    - last_conflict_year_<y_end>.csv: last conflict year per polygon (decay).
    """
    y_end = config["general"]["y_end"]
    XY_df = cached["XY_df"]
    mask = (XY_df["year"] == y_end) & (XY_df["conflict"] == 1)
    ids_with_conflict = pd.unique(XY_df.loc[mask, "poly_ID"])
    files_dir = os.path.join(out_dir, "files")
    os.makedirs(files_dir, exist_ok=True)
    bool_df = pd.DataFrame(
        {"bool_conflict": 1},
        index=pd.Index(ids_with_conflict, name="watprovID"),
    )
    out_file = os.path.join(files_dir, f"conflicts_in_{y_end}.csv")
    click.echo(f"Regenerating {out_file} from cache for projections.")
    bool_df.to_csv(out_file)

    # last conflict year per polygon (decay initial conditions); derivable
    # from the cached conflict column itself
    conf_rows = XY_df.loc[XY_df["conflict"] == 1, ["poly_ID", "year"]]
    last_df = (
        conf_rows.groupby("poly_ID")["year"]
        .max()
        .astype(int)
        .to_frame("last_conflict_year")
    )
    last_df.index = last_df.index.rename("watprovID")
    out_file = os.path.join(files_dir, f"last_conflict_year_{y_end}.csv")
    click.echo(f"Regenerating {out_file} from cache for projections.")
    last_df.to_csv(out_file)


def _fill_XY(  # noqa: R0912
    XY: dict,
    config: dict,
    root_dir: click.Path,
    conflict_data: gpd.GeoDataFrame,
    # target_var: Union[str, None],
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

    # Pre-compute the conflict-history lookups ONCE (a single spatial join
    # each) instead of re-joining inside the year loop.
    conflict_years = conflict.conflict_years_lookup(conflict_data, polygon_gdf)

    years_list = []
    
    for (sim_year, i) in zip(model_period, range(len(model_period))):

        if i == 0:
            click.echo(f"Skipping first year {sim_year} to start up model.")
        else:
            click.echo(f"Entering year {sim_year}.")
            years_list.extend([sim_year] * len(polygon_gdf))
            # go through all keys in dictionary
            for key, value in XY.items():

                if key == "conflict":

                    data_series = value
                    # TODO: guess for target_vars others than None, a dedicasted function is needed
                    # if target_var is None:
                    #     data_list = conflict.conflict_in_year_bool(
                    #         config, conflict_data, polygon_gdf, sim_year, out_dir
                    #     )
                    # else:
                    #     raise NotImplementedError(
                    #         "Implementation of target_var did not happen yet."
                    #     )
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

                elif key in ["conflict_decay_hl2", "conflict_decay_hl8"]:

                    half_life = 2.0 if key == "conflict_decay_hl2" else 8.0
                    data_list = conflict.decay_in_year(
                        polygon_gdf, conflict_years, sim_year, half_life
                    )
                    XY[key] = pd.concat(
                        [value, pd.Series(data_list)], axis=0, ignore_index=True
                    )

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

    # projection initial conditions for the conflict-history features,
    # written next to conflicts_in_<y_end>.csv
    conflict.store_conflict_history_files(
        conflict_years,
        polygon_gdf,
        config["general"]["y_end"],
        out_dir,
    )

    XY_df = pd.DataFrame.from_dict(XY)
    XY_df.insert(0, "year", years_list)
    return XY_df


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

def _nan_counts(df: pd.DataFrame) -> pd.Series:
    """Missing-value count per column, descending, columns with none omitted."""
    counts = df.isna().sum()
    return counts[counts > 0].sort_values(ascending=False)
 
 
def _describe_dropped(XY_df: pd.DataFrame, XY_df_noNaNs: pd.DataFrame) -> list:
    """Accounts for what dropna() removed, as a list of printable lines.
 
    Reports three things: columns that are entirely NaN (each one is on its own
    sufficient to empty the matrix), columns with partial gaps, and polygons
    that lost every row - with, for each of those, the columns responsible.
 
    Args:
        XY_df (pd.DataFrame): the matrix before dropna().
        XY_df_noNaNs (pd.DataFrame): the matrix after dropna().
 
    Returns:
        list: lines to print. Empty if nothing was missing.
    """
 
    n_total = len(XY_df)
    counts = _nan_counts(XY_df)
    if counts.empty:
        return []
 
    lines = []
 
    all_nan = [c for c in counts.index if counts[c] == n_total]
    if all_nan:
        lines.append(
            f"  ENTIRELY NaN ({n_total}/{n_total} rows): {', '.join(map(str, all_nan))}"
        )
 
    partial = [c for c in counts.index if counts[c] < n_total]
    if partial:
        lines.append(
            "  partial gaps: "
            + ", ".join(f"{c} {counts[c]}" for c in partial)
        )
 
    # polygons that lost every one of their rows
    ids_before = set(XY_df["poly_ID"].unique())
    ids_after = (
        set(XY_df_noNaNs["poly_ID"].unique()) if len(XY_df_noNaNs) else set()
    )
    lost = sorted(ids_before - ids_after, key=str)
    if lost:
        shown = ", ".join(map(str, lost[:DROP_REPORT_MAX_IDS]))
        more = (
            ""
            if len(lost) <= DROP_REPORT_MAX_IDS
            else f" (+{len(lost) - DROP_REPORT_MAX_IDS} more)"
        )
        lines.append(
            f"  polygons dropped entirely ({len(lost)} of {len(ids_before)}): "
            f"{shown}{more}"
        )
        # geometry is never the reason and would only add noise
        cols = [c for c in XY_df.columns if c != "poly_geometry"]
        for pid in lost[:DROP_REPORT_MAX_IDS]:
            sub = XY_df.loc[XY_df["poly_ID"] == pid, cols]
            bad = [c for c in cols if sub[c].isna().any()]
            lines.append(f"    {pid}: {', '.join(map(str, bad))}")
 
    return lines

def _split_XY_data(
    XY_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separates the XY-array into array containing information about
    variable values (X-array or sample data) and conflict data (Y-array or target data).
    Thereby, the X-array also contains the information about
    unique identifier and polygon geometry.
 
    Rows with any missing value are dropped, as before. What is new is that the
    drop is now accounted for: an empty result raises with the offending columns
    named, and a material partial loss is reported with the columns and polygons
    involved. See DROP_WARN_ROW_FRACTION / DROP_REPORT_MAX_IDS.
 
    Args:
        XY_df (pd.DataFrame): array containing variable values and conflict data.
 
    Returns:
        pd.DataFrame: X-array, i.e. array containing feature values.
        pd.DataFrame: Y-array, i.e. array containing target values.
        pd.DataFrame: meta-array used to stratify predictions.
 
    Raises:
        ValueError: if no rows survive the drop.
    """
 
    # drop missing values
    XY_df_noNaNs = XY_df.dropna().reset_index(drop=True)
 
    n_total = len(XY_df)
    n_kept = len(XY_df_noNaNs)
    n_dropped = n_total - n_kept
    polys_before = XY_df.poly_ID.nunique()
    polys_after = XY_df_noNaNs.poly_ID.nunique()
 
    click.echo(
        f"Dropped missing values, which leaves {n_kept} out of {n_total} data points."
        f"Number of polygons with data is now {polys_after} out of {polys_before}."
    )
 
    # --- nothing survived: fail with the reason, not a ZeroDivisionError -----
    if n_kept == 0:
        msg = [
            f"All {n_total} rows were dropped - every row has at least one "
            "missing value, so there is nothing to train on.",
        ]
        msg += _describe_dropped(XY_df, XY_df_noNaNs)
        msg += [
            "",
            "A single entirely-NaN indicator is enough to do this, because "
            "dropna() is row-wise.",
            "Such an indicator can arrive without any config change: INDICATORS "
            "is built by intersecting indicator_categories.csv with the "
            "*_global.nc files present on disk, so a stray or half-written file "
            "joins the feature set silently.",
            "If you repair an indicator in place rather than removing it, DELETE "
            "THE XY CACHE before re-running - _check_XY_cache validates the "
            "column list, years and input filenames, never the file contents.",
        ]
        raise ValueError("\n".join(msg))
 
    # --- partial loss: succeed, but say what went missing --------------------
    if n_dropped > 0:
        fraction = n_dropped / n_total
        if fraction > DROP_WARN_ROW_FRACTION or polys_after < polys_before:
            click.echo(
                click.style(
                    f"WARNING: {n_dropped} of {n_total} rows "
                    f"({100 * fraction:.1f}%) and "
                    f"{polys_before - polys_after} of {polys_before} polygons "
                    "were dropped for missing values. Dropped polygons are "
                    "absent from the evaluation, the projections and every map, "
                    "and nothing downstream will flag them.",
                    fg="red",
                )
            )
            for line in _describe_dropped(XY_df, XY_df_noNaNs):
                click.echo(click.style(line, fg="red"))
 
    # ride-along metadata: never enters the model, used to stratify predictions.
    # year -> temporal split; conflict_t_min_1 -> emerging vs ongoing.
    # poly_ID is kept only to assert alignment after the train/test split.
    meta_df = XY_df_noNaNs[["year", "poly_ID", "conflict_t_min_1"]].copy()
 
    # X data: drop year (metadata, not a feature) and conflict (target, last col).
    # Leaves poly_ID, poly_geometry, indicators, and the two conflict_t_min_1
    # columns - identical to the X structure before this change.
    X_df = XY_df_noNaNs.drop(columns=["year"]).iloc[:, :-1]
 
    # Y data, as integers
    Y_df = XY_df_noNaNs.iloc[:, -1].astype(int)
 
    fraction_Y_1 = 100 * len(Y_df[Y_df == 1]) / len(Y_df)
    click.echo(
        f"{round(fraction_Y_1, 2)} percent in the data corresponds to conflicts."
    )
 
    return X_df, Y_df, meta_df