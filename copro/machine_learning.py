import os
import pickle
import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing, model_selection, inspection
from typing import Union, Tuple
import click
from pathlib import Path
from sklearn.model_selection import GridSearchCV, KFold


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
        self, X: Union[np.ndarray, pd.DataFrame], Y: np.ndarray
    ):
        """Splits and transforms the X-array (or sample data) and
        Y-array (or target data) in test-data and training-data.
        The fraction of data used to split the data is specified in the configuration file.
        Additionally, the unique identifier and geometry of each data point in both
        test-data and training-data is retrieved in separate arrays.

        Args:
            X (array): array containing the variable values plus unique identifer and geometry information.
            Y (array): array containing merely the binary conflict classifier data.

        Returns:
            arrays: arrays containing training-set and test-set for X-data and Y-data as well as IDs and geometry.
        """

        ##- separate arrays for ID, geometry, and variable values
        X_ID, X_geom, X_data = _split_conflict_geom_data(X)

        ##- scaling only the variable values
        click.echo("Fitting and transforming scaler.")
        X_ft = self.scaler.fit_transform(X_data)

        ##- combining ID, geometry and scaled sample values per polygon
        X_cs = np.column_stack((X_ID, X_geom, X_ft))

        ##- splitting in train and test samples based on user-specified fraction
        click.echo("Splitting both X and Y in train and test data.")
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X_cs,
            Y,
            test_size=1 - self.config["machine_learning"]["train_fraction"],
        )

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
        )

    def fit_predict(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: np.ndarray,
        X_test: Union[np.ndarray, pd.DataFrame],
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
            X_test (np.ndarray, pd.DataFrame): test-data of variable values.
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
            fitted_estimator = apply_gridsearchCV(
                self.estimator, X_train, y_train, n_jobs=n_jobs, verbose=verbose
            )
        else:
            # fit the classifier with training data
            fitted_estimator = self.estimator.fit(X_train, y_train)

        # compute permutation importance
        click.echo("Computing permutation importance.")
        perm_importances = inspection.permutation_importance(
            fitted_estimator,
            X_train,
            y_train,
            n_repeats=10,
            random_state=42,
            n_jobs=n_jobs,
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
    """Loads the paths to all previously fitted classifiers to a list.
    Classifiers were saved to file in fit_predict().
    With this list, the classifiers can be loaded again during projections.

    Args:
        config (dict): Parsed configuration-settings of the model.
        out_dir (path): path to output folder.

    Returns:
        list: list with file names of classifiers.
    """

    estimators = os.listdir(os.path.join(out_dir, "estimators"))

    if len(estimators) != config["machine_learning"]["n_runs"]:
        raise ValueError(
            "Number of loaded classifiers does not match the specified number of runs in cfg-file!"
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
    n_jobs=2,
    verbose=0,
) -> Union[ensemble.RandomForestClassifier, ensemble.RandomForestRegressor]:
    """Applies grid search to find the best hyperparameters for the RandomForestClassifier.

    Args:
        estimator (Union[RandomForestClassifier, RandomForestRegressor]): Estimator to be used in the grid search.
        X_train (np.ndarray): Feature matrix.
        y_train (np.ndarray): Target vector.
        n_jobs (int, optional): Number of cores to be used. Defaults to 2.
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        Union[ensemble.RandomForestClassifier, ensemble.RandomForestRegressor]: Best estimator of the grid search.
    """

    click.echo("Tuning hyperparameters with GridSearchCV.")
    # Define the parameter grid
    if isinstance(estimator, ensemble.RandomForestClassifier):
        param_grid = {
            "n_estimators": [50, 100, 200],
            "criterion": ["gini", "entropy"],
            "min_impurity_decrease": [0, 0.5, 1],
            "max_features": ("sqrt", "log2"),
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": [{1: 75}, {1: 100}, {1: 150}],
            # 'bootstrap': [True, False]
        }
        scoring = "roc_auc"
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

    # Instantiate the grid search model
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=KFold(n_splits=5, shuffle=True),
        n_jobs=n_jobs,
        verbose=verbose,
        scoring=scoring,
    )

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best estimator
    best_estimator = grid_search.best_estimator_
    click.echo(f"ROC-AUC of best estimator is {grid_search.best_score_}.")
    click.echo(f"Best estimator is {grid_search.best_estimator_}.")

    return best_estimator
