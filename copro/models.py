from copro import machine_learning, conflict, evaluation, utils, xydata
from configparser import RawConfigParser
from sklearn import ensemble
import pandas as pd
import numpy as np
from typing import Union
import geopandas as gpd
import click
import os
import pickle


class MainModel:
    def __init__(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        Y: np.ndarray,
        config: RawConfigParser,
        out_dir: str,
    ):
        """Constructor for the MainModel class.

        Args:
            X (np.ndarray, pd.DataFrame): array containing the variable values plus IDs and geometry information.
            Y (np.ndarray): array containing merely the binary conflict classifier data.
            config (ConfigParser-object): object containing the parsed configuration-settings of the model.
            scaler (scaler): the specified scaling method instance.
            clf (classifier): the specified model instance.
            out_dir (str): path to output folder.
            run_nr (int): number of the current run.
        """
        self.X = X
        self.Y = Y
        self.config = config
        # TODO: scaler and clf settings need to be aligned with machine_learning.py class
        self.scaler = machine_learning.define_scaling(config)
        self.clf = ensemble.RandomForestClassifier(
            n_estimators=1000, class_weight={1: 100}, random_state=42
        )
        self.out_dir = out_dir

    def run(self, run_nr) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Main model workflow when all XY-data is used.
        The model workflow is executed for each classifier.

        Returns:
            dataframe: containing the test-data X-array values.
            datatrame: containing model output on polygon-basis.
            dict: dictionary containing evaluation metrics per simulation.
        """

        MLmodel = machine_learning.MachineLearning(
            self.config,
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
        y_pred, y_prob = MLmodel.fit_predict(
            X_train, y_train, X_test, self.out_dir, run_nr
        )
        y_prob_0 = y_prob[:, 0]  # probability to predict 0
        y_prob_1 = y_prob[:, 1]  # probability to predict 1

        # evaluate prediction and save to dict
        eval_dict = evaluation.evaluate_prediction(y_test, y_pred, y_prob)

        # aggregate predictions per polygon
        y_df = conflict.check_for_correct_prediction(
            X_test_ID, X_test_geom, y_test, y_pred, y_prob_0, y_prob_1
        )

        return X_df, y_df, eval_dict

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

        clfs = machine_learning.load_clfs(config_REF, out_dir_REF)

        # initiate output dataframe
        all_y_df = pd.DataFrame(columns=["ID", "geometry", "y_pred"])

        # going through each projection specified
        for each_key, _ in config_REF.items("PROJ_files"):

            # get config-object and out-dir per projection
            click.echo(f"Loading config-object for projection run: {each_key}.")
            config_PROJ = main_dict[str(each_key)][0][0]
            out_dir_PROJ = main_dict[str(each_key)][1]

            click.echo(f"Storing output for this projections to folder {out_dir_PROJ}.")

            # if not os.path.isdir(os.path.join(out_dir_PROJ, 'files')):
            #     os.makedirs(os.path.join(out_dir_PROJ, 'files'))
            if not os.path.isdir(os.path.join(out_dir_PROJ, "clfs")):
                os.makedirs(os.path.join(out_dir_PROJ, "clfs"))

            # get projection period for this projection
            # defined as all years starting from end of reference run until specified end of projections
            projection_period = utils.determine_projection_period(
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

                    X = xydata.fill_X_conflict(
                        X, config_PROJ, conflict_data, selected_polygons_gdf
                    )
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
                            X, config_PROJ, conflict_data, selected_polygons_gdf
                        )
                        X = pd.DataFrame.from_dict(X).to_numpy()

                    X = pd.DataFrame(X)
                    X = X.fillna(0)

                    # put all the data into the machine learning algo
                    # here the data will be used to make projections with various classifiers
                    # returns the prediction based on one individual classifier

                    # TODO: check what this function should contain
                    # TODO: and where it is now in the upcated copro code
                    # y_df_clf = models.predictive(X, clf_obj, self.scaler, config_PROJ)
                    y_df_clf = pd.DataFrame(clf_obj)

                    # storing the projection per clf to be used in the following timestep
                    y_df_clf.to_csv(
                        os.path.join(
                            out_dir_PROJ,
                            "clfs",
                            str(clf).rsplit(".", maxsplit=1)[0],
                            "projection_for_{}.csv".format(proj_year),
                        )
                    )

                    # removing projection of previous time step as not needed anymore
                    if i > 0:
                        os.remove(
                            os.path.join(
                                out_dir_PROJ,
                                "clfs",
                                str(clf).rsplit(".", maxsplit=1)[0],
                                "projection_for_{}.csv".format(proj_year - 1),
                            )
                        )

                    # append to all classifiers dataframe
                    y_df = y_df.append(y_df_clf, ignore_index=True)

                # get look-up dataframe to assign geometry to polygons via unique ID
                global_df = utils.global_ID_geom_info(selected_polygons_gdf)

                click.echo(
                    f"Storing model output for year {proj_year} to output folder".format
                )
                _, gdf_hit = evaluation.polygon_model_accuracy(
                    y_df, global_df, make_proj=True
                )
                gdf_hit.to_file(
                    os.path.join(out_dir_PROJ, f"output_in_{proj_year}.geojson"),
                    driver="GeoJSON",
                )

            # create one major output dataframe containing all output for all projections with all classifiers
            all_y_df = all_y_df.append(y_df, ignore_index=True)

        return all_y_df
