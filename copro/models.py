import json
import os
from pathlib import Path
from typing import Union

import click
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn import ensemble

from copro import conflict, evaluation, machine_learning, settings, utils, xydata


class MainModel:
    def __init__(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        Y: Union[np.ndarray, pd.DataFrame],
        meta: pd.DataFrame,
        estimator: Union[
            ensemble.RandomForestClassifier, ensemble.RandomForestRegressor
        ],
        config: dict,
        out_dir: str,
        n_jobs=2,
        verbose=0,
    ):
        """Constructor for the MainModel class.
        Under the hood, the class uses the `machine_learning.MachineLearning()` class to run the computations.

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
        self.meta = meta
        self.config = config
        self.estimator = estimator
        self.out_dir = out_dir
        self.n_jobs = n_jobs
        click.echo(f"Number of jobs to run in parallel: {n_jobs}.")
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
            X_train_ID,
            X_test_ID,
            meta_test,
        ) = MLmodel.split_scale_train_test_split(self.X, self.Y, self.meta, run_nr)
        
        # convert to dataframe
        X_df = pd.DataFrame(X_test)

        # fit classifier and make prediction with test-set
        y_pred, y_prob, perm_importances_arr_n = MLmodel.fit_predict(
            X_train,
            y_train,
            X_train_ID,
            X_test,
            y_test,
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

        # ride-along metadata is split in lockstep with X; assert it stayed aligned.
        # Compare as strings: watprovID may be non-numeric, and numeric coercion
        # would turn such IDs into NaN (NaN != NaN), falsely tripping this guard.
        ids_from_X = np.asarray(X_test_ID).astype(str)
        ids_from_meta = meta_test["poly_ID"].to_numpy().astype(str)
        if not np.array_equal(ids_from_X, ids_from_meta):
            click.echo(f"  X_test_ID head:    {ids_from_X[:10]}")
            click.echo(f"  meta poly_ID head: {ids_from_meta[:10]}")
            click.echo(f"  lengths: X={len(ids_from_X)}, meta={len(ids_from_meta)}")
            raise ValueError(
                "Ride-along metadata is misaligned with the test set "
                "(poly_ID mismatch) - check the train/test split."
            )

        y_df["run_nr"] = run_nr
        y_df["year"] = meta_test["year"].to_numpy()
        y_df["conflict_t_min_1"] = meta_test["conflict_t_min_1"].to_numpy()
        return X_df, y_df, eval_dict, perm_importances_arr_n


def run_prediction(
    main_dict: dict,
    root_dir: click.Path,
    selected_polygons_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Top-level function to execute the projections.
    Per projection, conflict is projected forward in time per time step until
    the projection year is reached. The sample (indicator) data is built once
    per year and shared by all classifiers; the conflict-at-t-1 data is applied
    per classifier, since each classifier produces its own conflict trajectory.

    Projections come from the DEPLOY model, which is fitted on all rows by
    `copro/scripts/deploy.py`, with the scaler that was fitted on those same
    rows. Projection from the evaluation run is retired: those estimators saw
    only the training portion, and under a temporal split that means the most
    recent conflict history was withheld from a model projecting forward.

    This is a module-level function rather than a `MainModel` method because it
    uses no instance state, and constructing a `MainModel` would mean running
    the evaluation fit first.

    Args:
        main_dict (dict): config-objects and output directories for the
            deploy model (key `_DEPLOY`) and all projection runs.
        root_dir (str): path to location of cfg-file.
        selected_polygons_gdf (geo-dataframe): polygons used in the run.

    Returns:
        dataframe: containing model output on polygon-basis.
    """

    config_DEPLOY = main_dict["_DEPLOY"][0]
    out_dir_DEPLOY = main_dict["_DEPLOY"][1]

    estimator, scaler, manifest = machine_learning.load_deploy_model(out_dir_DEPLOY)

    # Score-to-probability rules, written by copro/scripts/calibrate.py. They
    # are optional: without them the projection carries the score (y_prob_1)
    # only. If they are there they must be the rules of THIS deploy model: a
    # deploy fit that was redone leaves the old calibration.json in place.
    calibration_file = os.path.join(out_dir_DEPLOY, "calibration.json")
    calibration_content = None
    if os.path.isfile(calibration_file):
        with open(calibration_file, "r", encoding="utf-8") as f:
            calibration_content = json.load(f)
        if calibration_content["deploy_created_utc"] != manifest["created_utc"]:
            raise ValueError(
                f"{calibration_file} belongs to a deploy model fitted at "
                f"{calibration_content['deploy_created_utc']}, this one was fitted "
                f"at {manifest['created_utc']}. Run copro.scripts.calibrate again."
            )
        click.echo(
            f"Applying score-to-probability rules from {calibration_file} "
            f"(rows verified: {calibration_content['rows_verified']})."
        )
    else:
        click.echo(
            f"No calibration.json in {out_dir_DEPLOY}: projections carry the "
            "score (y_prob_1) only, no calibrated_probability."
        )

    # initiate output dataframe
    all_y_df = pd.DataFrame(columns=["ID", "geometry", "y_pred"])

    # Score only the polygons the deploy fit covered. Polygons dropped by the
    # dropna in _split_XY_data have no model, so scoring them would mean
    # scoring fabricated inputs. The manifest states the coverage, so it is the
    # same in every year and every scenario.
    n_selected = len(selected_polygons_gdf)
    selected_polygons_gdf = selected_polygons_gdf.loc[
        selected_polygons_gdf["watprovID"].isin(manifest["covered_poly_IDs"])
    ]
    click.echo(
        f"Projecting {len(selected_polygons_gdf)} of {n_selected} selected polygons; "
        f"{n_selected - len(selected_polygons_gdf)} are not covered by the deploy model."
    )

    # going through each projection specified
    for projection, _ in config_DEPLOY["projections"].items():

        click.echo(f"Loading config-object for projection run: {projection}.")
        config_PROJ = main_dict[str(projection)][0][0]
        out_dir_PROJ = main_dict[str(projection)][1]

        click.echo(f"Storing output for this projection to folder {out_dir_PROJ}.")
        Path.mkdir(
            Path(os.path.join(out_dir_PROJ, "estimators")),
            parents=True,
            exist_ok=True,
        )

        # projection period for this projection: from the reference end year
        # up to and including this projection's y_end
        projection_period = settings.determine_projection_period(
            config_DEPLOY, projection
        )

        # for this projection, go through all years
        for i, proj_year in enumerate(projection_period):

            click.echo(f"Making projection for year {proj_year}.")

            # Build the sample (indicator) data once per year. Kept as a dict
            # with string keys for the whole year so fill_X_conflict can match
            # its target columns; it is identical for every classifier.
            X_sample = xydata.initiate_X_data(config_PROJ)
            X_sample = xydata.fill_X_sample(
                X_sample, config_PROJ, root_dir, selected_polygons_gdf, proj_year
            )

            # For the first projection year, conflict at t-1 is the observed
            # conflict at the last reference year - shared by all classifiers.
            if i == 0:
                y_end_ref = config_DEPLOY["general"]["y_end"]
                ref_conflict_file = os.path.join(
                    out_dir_DEPLOY,
                    "files",
                    "conflicts_in_{}.csv".format(y_end_ref),
                )
                click.echo(
                    f"Reading previous conflicts from file {ref_conflict_file}."
                )
                observed_conflict = pd.read_csv(ref_conflict_file, index_col=0)
                # initial conditions for the conflict-history features
                last_obs_file = os.path.join(
                    out_dir_DEPLOY,
                    "files",
                    "last_conflict_year_{}.csv".format(y_end_ref),
                )
                click.echo(f"Reading last conflict years from file {last_obs_file}.")
                last_observed = pd.read_csv(last_obs_file, index_col=0)[
                    "last_conflict_year"
                ]

            # collect the projections from all classifiers for this timestep
            y_df = pd.DataFrame(columns=["ID", "geometry", "y_pred"])

            # One deploy estimator. The loop is kept because the per-estimator
            # trajectory folders below are what make the autoregression work,
            # and a random_state spread over deploy fits would slot in here
            # unchanged.
            for estimator_file in [manifest["estimator_file"]]:

                estimator_base = estimator_file.rsplit(".", maxsplit=1)[0]
                estimator_dir = os.path.join(
                    out_dir_PROJ, "estimators", estimator_base
                )
                os.makedirs(estimator_dir, exist_ok=True)

                # pick this classifier's conflict-at-t-1 source:
                # - first year: the shared observed conflict
                # - later years: THIS classifier's own previous-year projection
                if i == 0:
                    conflict_data = observed_conflict
                else:
                    prev_proj_file = os.path.join(
                        estimator_dir,
                        "projection_for_{}.csv".format(proj_year - 1),
                    )
                    click.echo(
                        f"Reading previous conflicts from file {prev_proj_file}."
                    )
                    # BUGFIX: this table is written by predictive() with a
                    # plain RangeIndex and one row per polygon. Reading it
                    # with index_col=0 gave an integer index, so the
                    # membership test in read_projected_conflict (which
                    # expects watprovIDs in the index, like
                    # conflicts_in_<y_end>.csv) matched nothing - conflict
                    # at t-1 was identically 0 for every polygon from the
                    # second projection year onwards. Filter to projected
                    # conflict polygons and index by ID to restore the
                    # conflicts_in_* convention.
                    conflict_data = pd.read_csv(prev_proj_file)
                    conflict_data = conflict_data.loc[
                        conflict_data["y_pred"] == 1
                    ].set_index("ID")

                # Last conflict year per polygon for the decay features:
                # observed history (<= y_end), overridden by any later
                # projected conflict years from THIS classifier's own
                # stored trajectory (the same files that feed conflict at
                # t-1, read with the same y_pred == 1 convention).
                last_lookup = last_observed.to_dict()
                for past_year in range(y_end_ref + 1, proj_year):
                    past_file = os.path.join(
                        estimator_dir,
                        "projection_for_{}.csv".format(past_year),
                    )
                    past = pd.read_csv(past_file)
                    for p in past.loc[past["y_pred"] == 1, "ID"]:
                        last_lookup[p] = past_year

                # Fresh per-classifier copy of the sample dict, so the string
                # keys stay intact and each classifier gets its own conflict at
                # t-1 (rather than leaking the previous classifier's state).
                X_est = dict(X_sample)
                X_est = xydata.fill_X_conflict(
                    X_est,
                    conflict_data,
                    selected_polygons_gdf,
                    proj_year,
                    last_lookup,
                )
                X_est = pd.DataFrame.from_dict(X_est)

                # Neither the estimator nor the scaler carries feature names,
                # so sklearn can only check the column count. This is the only
                # place an order or set mismatch can be caught.
                machine_learning.check_projection_columns(manifest, X_est)

                # fillna(0) used to sit here. Zero is not a neutral value:
                # QuantileTransformer clips it to the bottom of the fitted
                # range for every level feature, so a gap silently became
                # "least developed", "poorest" or "coldest" depending on which
                # way the feature happens to point. Every polygon scored here
                # was fitted, so a gap means an input layer is incomplete for
                # this year. predict() would raise on the NaN anyway, but
                # without naming the feature or the polygon, which is the whole
                # diagnosis.
                gap_cols = X_est.columns[X_est.isna().any()].tolist()
                if gap_cols:
                    gap_polys = X_est.loc[
                        X_est.isna().any(axis=1), "poly_ID"
                    ].tolist()
                    raise ValueError(
                        f"Missing values in the projection matrix for {proj_year}.\n"
                        f"  features with gaps: {gap_cols}\n"
                        f"  polygons affected:  {gap_polys}\n"
                        "These polygons were fitted, so this is an incomplete "
                        "input layer for this year, not a row to drop."
                    )

                # the matrix exactly as scored, raw values, for attribution
                # (copro/scripts/explain.py)
                X_est[["poly_ID"] + manifest["feature_columns"]].to_csv(
                    os.path.join(estimator_dir, "X_for_{}.csv".format(proj_year)),
                    index=False,
                )

                # project conflict risk with this classifier
                y_df_clf = machine_learning.predictive(X_est, estimator, scaler)

                # The probability beside the score. y_prob_1 is left as it is:
                # ranking and explain.py work on the score. The stratum is the
                # conflict-at-t-1 value the model itself was given, so in the
                # first projection year it is observed and in later years it is
                # this classifier's own previous-year call.
                if calibration_content is not None:
                    y_df_clf["stratum"] = machine_learning.strata_from_conflict(
                        X_est["conflict_t_min_1"]
                    )
                    y_df_clf["calibrated_probability"] = machine_learning.score_to_probability(
                        calibration_content["rules"],
                        y_df_clf["stratum"].to_numpy(),
                        pd.to_numeric(y_df_clf["y_prob_1"]).to_numpy(),
                    )

                # store this classifier's projection to feed the next timestep
                y_df_clf.to_csv(
                    os.path.join(
                        estimator_dir,
                        "projection_for_{}.csv".format(proj_year),
                    )
                )

                y_df = pd.concat([y_df, y_df_clf], axis=0, ignore_index=True)

            # assign geometry to polygons via unique ID and store year output
            global_df = utils.get_ID_geometry_lookup(selected_polygons_gdf)
            click.echo(f"Storing model output for year {proj_year} to output folder.")
            gdf_hit = evaluation.polygon_model_accuracy(y_df, global_df, make_proj=True)
            gdf_hit.to_file(
                os.path.join(out_dir_PROJ, f"output_in_{proj_year}.gpkg"),
                driver="GPKG",
                crs="EPSG:4326",
            )

        # combine all classifier output for this projection
        all_y_df = pd.concat([all_y_df, y_df], axis=0, ignore_index=True)

    return all_y_df
