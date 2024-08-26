from copro import machine_learning, conflict, evaluation, utils, xydata, settings
from configparser import RawConfigParser
from sklearn import ensemble
from sklearn.utils.validation import check_is_fitted
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union, Tuple
import geopandas as gpd
import click
import os
import pickle


class MainModel:
    def __init__(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        Y: np.ndarray,
        estimator: Union[
            ensemble.RandomForestClassifier, ensemble.RandomForestRegressor
        ],
        config: dict,
        out_dir: str,
        n_jobs=2,
        verbose=0,
    ):
        """Constructor for the MainModel class.

        Args:
            X (np.ndarray, pd.DataFrame): array containing the variable values plus IDs and geometry information.
            Y (np.ndarray): array containing merely the binary conflict classifier data.
            estimator (Union[ensemble.RandomForestClassifier, ensemble.RandomForestRegressor]): ML model.
            config (dict): object containing the parsed configuration-settings of the model.
            out_dir (str): path to output folder.
            n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 2.
            verbose (int, optional): Verbosity level. Defaults to 0.
        """
        self.X = X
        self.Y = Y
        self.config = config
        self.scaler = machine_learning.define_scaling(config)
        self.scaler_all_data = self.scaler.fit(
            X[:, 2:]
        )  # NOTE: supposed to be used in projections
        self.estimator = estimator
        self.out_dir = out_dir
        self.n_jobs = n_jobs
        self.verbose = verbose

    def run(
        self, number_runs: int, tune_hyperparameters=False
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        """Top-level function to execute the machine learning model for all specified runs.

        Args:
            number_runs (int): Number of runs as specified in the settings-file.
            tune_hyperparameters (bool, optional): Whether to tune hyperparameters or not. Defaults to False.

        Returns:
            pd.DataFrame: Prediction dataframes.
            pd.DataFrame: model output on polygon-basis.
            np.ndarray: containing permutation importances for all runs.
            dict: evaluation dictionary.
        """

        check_is_fitted(self.scaler)

        # - initializing output variables
        out_X_df = pd.DataFrame()
        out_y_df = pd.DataFrame()
        out_perm_importances_arr = np.array([]).reshape(0, self.X.shape[1] - 2)
        out_dict = evaluation.init_out_dict()

        click.echo("Training and testing machine learning model")
        for n in range(number_runs):
            click.echo(f"Run {n+1} of {number_runs}.")

            # - run machine learning model and return outputs
            X_df, y_df, eval_dict, perm_importances_arr_n = self._n_run(
                run_nr=n, tune_hyperparameters=tune_hyperparameters
            )

            # - append per model execution
            out_X_df = pd.concat([out_X_df, X_df], axis=0, ignore_index=True)
            out_y_df = pd.concat([out_y_df, y_df], axis=0, ignore_index=True)
            out_perm_importances_arr = np.vstack(
                [out_perm_importances_arr, perm_importances_arr_n]
            )
            out_dict = evaluation.fill_out_dict(out_dict, eval_dict)

        return out_X_df, out_y_df, out_perm_importances_arr, out_dict

    def _n_run(
        self, run_nr: int, tune_hyperparameters=False
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict, np.ndarray]:
        """Runs workflow per specified number of runs.
        The model workflow is executed for each classifier.

        Args:
            run_nr (int): Number of run.
            tune_hyperparameters (bool, optional): Whether to tune hyperparameters or not. Defaults to False.

        Returns:
            pd.DataFrame: containing the test-data X-array values.
            pd.DataFrame: containing model output on polygon-basis.
            dict: dictionary containing evaluation metrics per simulation.
            np.ndarray: containing permutation importances for run n.
        """

        MLmodel = machine_learning.MachineLearning(
            self.config,
            self.estimator,
        )

        # split X into training-set and test-set, scale training-set data
        (
            X_train,
            X_test,
            y_train,
            y_test,
            _,
            X_test_geom,
            _,
            X_test_ID,
        ) = MLmodel.split_scale_train_test_split(self.X, self.Y)

        # convert to dataframe
        X_df = pd.DataFrame(X_test)

        # fit classifier and make prediction with test-set
        y_pred, y_prob, perm_importances_arr_n = MLmodel.fit_predict(
            X_train,
            y_train,
            X_test,
            self.out_dir,
            run_nr,
            tune_hyperparameters,
            self.n_jobs,
            self.verbose,
        )
        y_prob_0 = y_prob[:, 0]  # probability to predict 0
        y_prob_1 = y_prob[:, 1]  # probability to predict 1

        # evaluate prediction and save to dict
        eval_dict = evaluation.evaluate_prediction(y_test, y_pred, y_prob)

        # aggregate predictions per polygon
        y_df = conflict.check_for_correct_prediction(
            X_test_ID, X_test_geom, y_test, y_pred, y_prob_0, y_prob_1
        )

        return X_df, y_df, eval_dict, perm_importances_arr_n

    def run_prediction(
        self,
        main_dict: dict,
        root_dir: click.Path,
        selected_polygons_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """Top-level function to execute the projections.
        Per specified projection, conflict is projected forwards in time per time step 
        until the projection year is reached.
        Pear time step, the sample data and conflict data are read individually since different 
        conflict projections are made per classifier used.
        At the end of each time step, the projections of all classifiers are combined and output metrics determined.

        Args:
            main_dict (dict): dictionary containing config-objects and output directories \
                for reference run and all projection runs.
            root_dir (str): path to location of cfg-file.
            selected_polygons_gdf (geo-dataframe):

        Returns:
            dataframe: containing model output on polygon-basis.
        """

        config_REF = main_dict["_REF"][0]
        out_dir_REF = main_dict["_REF"][1]

        clfs, all_y_df = _init_prediction_run(config_REF, out_dir_REF)

        # going through each projection specified
        for each_key, _ in config_REF.items():

            # get config-object and out-dir per projection
            click.echo(f"Loading config-object for projection run: {each_key}.")
            config_PROJ = main_dict[str(each_key)][0][0]
            out_dir_PROJ = main_dict[str(each_key)][1]

            click.echo(f"Storing output for this projections to folder {out_dir_PROJ}.")
            Path.mkdir(
                Path(os.path.join(out_dir_PROJ, "clfs")), parents=True, exist_ok=True
            )

            # get projection period for this projection
            # defined as all years starting from end of reference run until specified end of projections
            projection_period = settings.determine_projection_period(
                config_REF, config_PROJ
            )

            # for this projection, go through all years
            for i, proj_year in enumerate(projection_period):

                click.echo(f"Making projection for year {proj_year}.")

                X = xydata.initiate_X_data(config_PROJ)
                X = xydata.fill_X_sample(
                    X, config_PROJ, root_dir, selected_polygons_gdf, proj_year
                )

                # for the first projection year, we need to fall back on the observed conflict
                # at the last time step of the reference run
                if i == 0:
                    click.echo(
                        "Reading previous conflicts from file {}".format(
                            os.path.join(
                                out_dir_REF,
                                "files",
                                "conflicts_in_{}.csv".format(
                                    config_REF.getint("settings", "y_end")
                                ),
                            )
                        )
                    )
                    conflict_data = pd.read_csv(
                        os.path.join(
                            out_dir_REF,
                            "files",
                            "conflicts_in_{}.csv".format(
                                config_REF.getint("settings", "y_end")
                            ),
                        ),
                        index_col=0,
                    )

                    X = xydata.fill_X_conflict(X, conflict_data, selected_polygons_gdf)
                    X = pd.DataFrame.from_dict(X).to_numpy()

                # initiating dataframe containing all projections from all classifiers for this timestep
                y_df = pd.DataFrame(columns=["ID", "geometry", "y_pred"])

                # now load all classifiers created in the reference run
                for clf in clfs:

                    # creating an individual output folder per classifier
                    if not os.path.isdir(
                        os.path.join(
                            os.path.join(
                                out_dir_PROJ,
                                "clfs",
                                str(clf).rsplit(".", maxsplit=1)[0],
                            )
                        )
                    ):
                        os.makedirs(
                            os.path.join(
                                out_dir_PROJ,
                                "clfs",
                                str(clf).rsplit(".", maxsplit=1)[0],
                            )
                        )

                    # load the pickled objects
                    # TODO: keep them in memory, i.e. after reading the clfs-folder above
                    with open(os.path.join(out_dir_REF, "clfs", clf), "rb") as f:
                        click.echo(
                            "Loading classifier {} from {}".format(
                                clf, os.path.join(out_dir_REF, "clfs")
                            )
                        )
                        clf_obj = pickle.load(f)

                    # for all other projection years than the first one,
                    # we need to read projected conflict from the previous projection year
                    if i > 0:
                        click.echo(
                            "Reading previous conflicts from file {}".format(
                                os.path.join(
                                    out_dir_PROJ,
                                    "clfs",
                                    str(clf),
                                    "projection_for_{}.csv".format(proj_year - 1),
                                )
                            )
                        )
                        conflict_data = pd.read_csv(
                            os.path.join(
                                out_dir_PROJ,
                                "clfs",
                                str(clf).rsplit(".", maxsplit=1)[0],
                                "projection_for_{}.csv".format(proj_year - 1),
                            ),
                            index_col=0,
                        )

                        X = xydata.fill_X_conflict(
                            X, conflict_data, selected_polygons_gdf
                        )
                        X = pd.DataFrame.from_dict(X).to_numpy()

                    X = pd.DataFrame(X)
                    X = X.fillna(0)

                    # put all the data into the machine learning algo
                    # here the data will be used to make projections with various classifiers
                    # returns the prediction based on one individual classifier
                    y_df_clf = machine_learning.predictive(
                        X, clf_obj, self.scaler_all_data
                    )

                    # storing the projection per clf to be used in the following timestep
                    y_df_clf.to_csv(
                        os.path.join(
                            out_dir_PROJ,
                            "clfs",
                            str(clf).rsplit(".", maxsplit=1)[0],
                            "projection_for_{}.csv".format(proj_year),
                        )
                    )

                    # append to all classifiers dataframe
                    y_df = pd.concat([y_df, y_df_clf], axis=0, ignore_index=True)

                # get look-up dataframe to assign geometry to polygons via unique ID
                global_df = utils.get_ID_geometry_lookup(selected_polygons_gdf)

                click.echo(
                    f"Storing model output for year {proj_year} to output folder."
                )
                gdf_hit = evaluation.polygon_model_accuracy(
                    y_df, global_df, make_proj=True
                )
                gdf_hit.to_file(
                    os.path.join(out_dir_PROJ, f"output_in_{proj_year}.geojson"),
                    driver="GeoJSON",
                )

            # create one major output dataframe containing all output for all projections with all classifiers
            all_y_df = pd.concat([all_y_df, y_df], axis=0, ignore_index=True)

        return all_y_df


def _init_prediction_run(
    config_REF: RawConfigParser, out_dir_REF: str
) -> Tuple[list, pd.DataFrame]:
    """Initializes the prediction run by loading all classifiers created in the reference run.
    Also initiates an empty dataframe to store the predictions.

    Args:
        config_REF (RawConfigParser): Reference configuration object.
        out_dir_REF (str): Output directory for reference run.

    Returns:
        Tuple[list, pd.DataFrame]: List with classifiers and initiated empty dataframe for predictions.
    """

    clfs = machine_learning.load_clfs(config_REF, out_dir_REF)

    # initiate output dataframe
    all_y_df = pd.DataFrame(columns=["ID", "geometry", "y_pred"])

    return clfs, all_y_df
