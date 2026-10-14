import os
import pickle
from pathlib import Path
from typing import Tuple, Union

import click
import numpy as np
import pandas as pd
from sklearn import ensemble, inspection, model_selection, preprocessing
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold


class MachineLearning:
    def __init__(
        self,
        config: dict,
        estimator: Union[
            ensemble.RandomForestClassifier, ensemble.RandomForestRegressor
        ],
    ) -> None:
        """Class for all ML related stuff.
        Embedded in more top-level `models.MainModel()` class.

        Args:
            config (dict): Parsed configuration-settings of the model.
            estimator (Union[ ensemble.RandomForestClassifier, ensemble.RandomForestRegressor ]): ML model.
        """
        self.config = config
        self.scaler = define_scaling(config)
        self.estimator = estimator

    def split_scale_train_test_split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        Y: np.ndarray,
        meta: pd.DataFrame,
        run_nr: int = 0,
    ):
        """Partitions X/Y/meta into train and test sets and scales the features.

        The partition strategy is read from the config and is one of:
          - "random": random row-wise hold-out (optimistic; the same polygon can
            appear in train and test across years).
          - "temporal": train on years <= temporal_cutoff, test on later years
            (out-of-time generalization).
          - "leave_polygons_out": whole polygons held out, so no polygon is in
            both train and test (spatial generalization; breaks the
            same-polygon-across-years leakage).

        The scaler is fit on the TRAINING rows only and then applied to the test
        rows, so no test-set information leaks into scaling. ID and geometry ride
        along unscaled; meta_test is returned for downstream stratification.
        """
        ##- separate arrays for ID, geometry, and variable values
        X_ID, X_geom, X_data = _split_conflict_geom_data(X)

        Y = np.asarray(Y)
        meta = meta.reset_index(drop=True)

        ml = self.config["machine_learning"]
        strategy = ml.get("split_strategy", "random")
        # vary the seed per run so n_runs > 1 produces different random/spatial
        # splits (reproducible given the base seed); temporal is deterministic
        # and unaffected.
        seed = ml.get("split_seed", 42) + run_nr
        test_fraction = 1 - ml["train_fraction"]
        n = X_data.shape[0]

        ##- determine train/test row indices according to the chosen strategy
        if strategy == "random":
            train_idx, test_idx = model_selection.train_test_split(
                np.arange(n), test_size=test_fraction, random_state=seed
            )
        elif strategy == "temporal":
            cutoff = ml.get("temporal_cutoff")
            if cutoff is None:
                raise ValueError(
                    "split_strategy='temporal' requires 'temporal_cutoff' "
                    "in the machine_learning config."
                )
            years = meta["year"].to_numpy()
            train_idx = np.where(years <= cutoff)[0]
            test_idx = np.where(years > cutoff)[0]
            if len(train_idx) == 0 or len(test_idx) == 0:
                raise ValueError(
                    f"temporal_cutoff={cutoff} leaves an empty train or test set."
                )
        elif strategy == "leave_polygons_out":
            groups = meta["poly_ID"].to_numpy()
            gss = model_selection.GroupShuffleSplit(
                n_splits=1, test_size=test_fraction, random_state=seed
            )
            train_idx, test_idx = next(gss.split(np.arange(n), Y, groups))
        else:
            raise ValueError(
                f"Unknown split_strategy '{strategy}'. Use 'random', "
                "'temporal', or 'leave_polygons_out'."
            )

        click.echo(
            f"Split strategy '{strategy}' (seed {seed}): "
            f"train={len(train_idx)}, test={len(test_idx)}."
        )

        ##- fit scaler on TRAINING data only, then transform both sets
        click.echo("Fitting scaler on training data and transforming both sets.")
        X_train_ft = self.scaler.fit_transform(X_data[train_idx])
        X_test_ft = self.scaler.transform(X_data[test_idx])

        ##- reattach unscaled ID and geometry
        X_train = np.column_stack((X_ID[train_idx], X_geom[train_idx], X_train_ft))
        X_test = np.column_stack((X_ID[test_idx], X_geom[test_idx], X_test_ft))

        y_train = Y[train_idx]
        y_test = Y[test_idx]
        meta_test = meta.iloc[test_idx]

        # for training-set and test-set, split in ID, geometry, and values
        X_train_ID, X_train_geom, X_train = _split_conflict_geom_data(X_train)
        X_test_ID, X_test_geom, X_test = _split_conflict_geom_data(X_test)

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            X_train_geom,
            X_test_geom,
            X_train_ID,
            X_test_ID,
            meta_test,
        )

    def fit_predict(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: np.ndarray,
        train_ID: np.ndarray,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: np.ndarray,
        out_dir: str,
        run_nr: int,
        tune_hyperparameters=False,
        n_jobs=2,
        verbose=0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fits classifier based on training-data and makes predictions.
        The fitted classifier is dumped to file with pickle to be used again during projections.
        Makes prediction with test-data including probabilities of those predictions.
        If specified, hyperparameters of classifier are tuned with GridSearchCV.

        Args:
            X_train (np.ndarray, pd.DataFrame): training-data of variable values.
            y_train (np.ndarray): training-data of conflict data.
            train_ID (np.ndarray): polygon ID per training row, used to group the inner CV folds.
            X_test (np.ndarray, pd.DataFrame): test-data of variable values.
            y_test (np.ndarray): test-data of conflict data, used for permutation importance.
            out_dir (str): path to output folder.
            run_nr (int): number of fit/predict repetition and created classifier.
            tune_hyperparameters (bool, optional): whether to tune hyperparameters. Defaults to False.
            n_jobs (int, optional): Number of cores to be used. Defaults to 2.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            np.ndarray: array with the predictions made.
            np.ndarray: array with probabilities of the predictions made.
            np.ndarray: dataframe containing permutation importances of variables.
        """

        if tune_hyperparameters:
            # run_nr varies the inner CV folds deterministically rather than
            # randomly; the base seed is the one the train/test split uses.
            cv_seed = self.config["machine_learning"].get("split_seed", 42) + run_nr
            fitted_estimator = apply_gridsearchCV(
                self.estimator,
                X_train,
                y_train,
                train_ID,
                n_jobs=n_jobs,
                verbose=verbose,
                seed=cv_seed,
            )
        else:
            # fit the classifier with training data
            fitted_estimator = self.estimator.fit(X_train, y_train)

        # compute permutation importance on the TEST set with ROC-AUC scoring.
        # Test set: avoids inflated importance from features the trees memorized
        # on training data. ROC-AUC: more sensitive than default accuracy for
        # imbalanced classification (rare conflict events), so importance values
        # are not compressed toward zero by the high accuracy baseline.
        click.echo("Computing permutation importance (test set, ROC-AUC).")
        perm_importances = inspection.permutation_importance(
            fitted_estimator,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=n_jobs,
            scoring="roc_auc",
        )
        # transpose because by default features are in rows
        perm_importances_arr = perm_importances["importances"].T

        # create folder to store all classifiers with pickle
        estimator_pickle_rep = os.path.join(out_dir, "estimators")
        Path.mkdir(Path(estimator_pickle_rep), parents=True, exist_ok=True)

        # save the fitted classifier to file via pickle.dump()
        click.echo(f"Dumping classifier to {estimator_pickle_rep}.")
        with open(
            os.path.join(estimator_pickle_rep, "estimator_{}.pkl".format(run_nr)), "wb"
        ) as f:
            pickle.dump(fitted_estimator, f)

        # make prediction
        y_pred = fitted_estimator.predict(X_test)
        # make prediction of probability
        y_prob = fitted_estimator.predict_proba(X_test)

        return y_pred, y_prob, perm_importances_arr


def load_estimators(config: dict, out_dir: str) -> list[str]:
    """Loads the paths to all previously fitted estimators to a list.
    Estimators were saved to file in `fit_predict()`.
    With this list, the classifiers can be loaded again during projections.

    Args:
        config (dict): Parsed configuration-settings of the model.
        out_dir (path): path to output folder.

    Returns:
        list: list with file names of estimators.
    """

    estimators = os.listdir(os.path.join(out_dir, "estimators"))

    if len(estimators) != config["machine_learning"]["n_runs"]:
        raise ValueError(
            "Number of loaded estimators does not match the specified number of runs in reference yaml-file!"
        )

    return estimators


def _split_conflict_geom_data(
    X: Union[np.ndarray, pd.DataFrame]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separates the unique identifier, geometry information, and data from the variable-containing X-array.

    Args:
        X (np.ndarray, pd.DataFrame): variable-containing X-array.

    Returns:
        arrays: seperate arrays with ID, geometry, and actual data
    """

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    # first column corresponds to ID, second to geometry
    # all remaining columns are actual data
    X_ID = X[:, 0]
    X_geom = X[:, 1]
    X_data = X[:, 2:]

    return X_ID, X_geom, X_data


def define_scaling(
    config: dict,
) -> Union[
    preprocessing.MinMaxScaler,
    preprocessing.StandardScaler,
    preprocessing.RobustScaler,
    preprocessing.QuantileTransformer,
]:
    """Defines scaling method based on model configurations.

    Args:
        config (dict): Parsed configuration-settings of the model.

    Returns:
        scaler: the specified scaling method instance.
    """

    if config["machine_learning"]["scaler"] == "MinMaxScaler":
        return preprocessing.MinMaxScaler()
    if config["machine_learning"]["scaler"] == "StandardScaler":
        return preprocessing.StandardScaler()
    if config["machine_learning"]["scaler"] == "RobustScaler":
        return preprocessing.RobustScaler()
    if config["machine_learning"]["scaler"] == "QuantileTransformer":
        return preprocessing.QuantileTransformer(random_state=42)

    raise ValueError(
        "no supported scaling-algorithm selected - \
            choose between MinMaxScaler, StandardScaler, RobustScaler or QuantileTransformer"
    )


def predictive(
    X: np.ndarray,
    estimator: ensemble.RandomForestClassifier,
    scaler: Union[
        preprocessing.MinMaxScaler,
        preprocessing.StandardScaler,
        preprocessing.RobustScaler,
        preprocessing.QuantileTransformer,
    ],
) -> pd.DataFrame:
    """Predictive model to use the already fitted classifier
    to make annual projections for the projection period.
    As other models, it reads data which are then scaled and
    used in conjuction with the classifier to project conflict risk.

    Args:
        X (np.ndarray): array containing the variable values plus unique identifer and geometry information.
        estimator (RandomForestClassifier): the fitted RandomForestClassifier.
        scaler (scaler): the fitted specified scaling method instance.

    Returns:
        pd.DataFrame: containing model output on polygon-basis.
    """

    # splitting the data from the ID and geometry part of X
    X_ID, X_geom, X_data = _split_conflict_geom_data(X.to_numpy())

    # transforming the data
    # fitting is not needed as already happend before
    X_ft = scaler.transform(X_data)

    # make projection with transformed data
    y_pred = estimator.predict(X_ft)

    # predict probabilites of outcomes
    y_prob = estimator.predict_proba(X_ft)
    y_prob_0 = y_prob[:, 0]  # probability to predict 0
    y_prob_1 = y_prob[:, 1]  # probability to predict 1

    # stack together ID, gemoetry, and projection per polygon, and convert to dataframe
    arr = np.column_stack((X_ID, X_geom, y_pred, y_prob_0, y_prob_1))
    y_df = pd.DataFrame(
        arr, columns=["ID", "geometry", "y_pred", "y_prob_0", "y_prob_1"]
    )

    return y_df


def apply_gridsearchCV(
    estimator: Union[ensemble.RandomForestClassifier, ensemble.RandomForestRegressor],
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    n_jobs=2,
    verbose=0,
    seed=42,
) -> Union[ensemble.RandomForestClassifier, ensemble.RandomForestRegressor]:
    """Applies grid search to find the best hyperparameters for the RandomForestClassifier.

    Args:
        estimator (Union[RandomForestClassifier, RandomForestRegressor]): Estimator to be used in the grid search.
        X_train (np.ndarray): Feature matrix.
        y_train (np.ndarray): Target vector.
        groups (np.ndarray): polygon ID per training row, used to keep whole polygons inside one inner fold.
        n_jobs (int, optional): Number of cores to be used. Defaults to 2.
        verbose (int, optional): Verbosity level. Defaults to 0.
        seed (int, optional): Seed for the inner CV fold assignment. Defaults to 42.


    Returns:
        Union[ensemble.RandomForestClassifier, ensemble.RandomForestRegressor]: Best estimator of the grid search.
    """

    click.echo("Tuning hyperparameters with GridSearchCV.")
    # Define the parameter grid
    if isinstance(estimator, ensemble.RandomForestClassifier):
        # Trimmed grid: 8 combinations (2 x 2 x 2) over the axes that matter
        # for rare-event conflict. The axes that are NOT searched are not
        # listed here: they arrive on the estimator from
        # settings.define_model (DEFAULT_HYPERPARAMETERS, or the config
        # block), which is their one definition:
        #   criterion             "gini"  (entropy rarely changes the split)
        #   min_impurity_decrease 0       (0.5/1 prune trees to near-stumps)
        #   max_features          "sqrt"  (standard RF default)
        #   min_samples_split     2       (default)
        #   bootstrap             True
        # The three axes below override whatever the estimator carries.
        param_grid = {
            "n_estimators": [100, 200],
            "min_samples_leaf": [1, 4],
            "class_weight": [{1: 100}, {1: 150}],
        }
        # average_precision (area under the precision-recall curve) selects
        # hyperparameters for the rare positive class, rather than roc_auc which
        # is padded by the large pool of easy true negatives.
        scoring = "average_precision"
        # Whole polygons stay inside one inner fold, so a configuration cannot
        # win by recognising which province a row belongs to. The outer
        # train/test split was already honest - the search only ever sees
        # X_train - but the inner fold that scores the eight candidates was
        # random over rows, which is the one place province identity could pay.
        # Stratified so the rare positive class stays proportionate across
        # folds: positives are spatially clustered, and plain GroupKFold
        # balances fold sizes only, which can leave a fold with very few
        # onsets and an unstable average_precision.
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)        
    else:
        param_grid = {
            "n_estimators": [10, 50, 100],
            "criterion": ("squared_error", "absolute_error", "friedman_mse"),
            "max_features": ("sqrt", "log2"),
            "min_samples_split": [2, 5, 20],
            "min_impurity_decrease": [0, 0.5, 1],
            "min_samples_leaf": [1, 5, 10],
        }
        scoring = "r2"
        # no stratification for a continuous target
        cv = GroupKFold(n_splits=3)
        
    # Instantiate the grid search model
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        scoring=scoring,
    )

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train, groups=groups)

    # Get the best estimator
    best_estimator = grid_search.best_estimator_
    click.echo(f"Best estimator {scoring} score is {grid_search.best_score_}.")
    click.echo(f"Best estimator is {grid_search.best_estimator_}.")

    return best_estimator
