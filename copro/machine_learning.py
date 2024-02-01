import os
import pickle
import pandas as pd
import numpy as np
from configparser import RawConfigParser
from sklearn import ensemble, preprocessing, model_selection
from copro import conflict
from typing import Union, Tuple


class MachineLearning:
    def __init__(self, config: RawConfigParser) -> None:
        self.config = config
        self.scaler = _define_scaling(config)
        self.clf = ensemble.RandomForestClassifier(
            n_estimators=1000, class_weight={1: 100}, random_state=42
        )

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
        X_ID, X_geom, X_data = conflict.split_conflict_geom_data(X)

        ##- scaling only the variable values
        if self.config.getboolean("general", "verbose"):
            print("DEBUG: fitting and transforming X")
        X_ft = self.scaler.fit_transform(X_data)

        ##- combining ID, geometry and scaled sample values per polygon
        X_cs = np.column_stack((X_ID, X_geom, X_ft))

        ##- splitting in train and test samples based on user-specified fraction
        if self.config.getboolean("general", "verbose"):
            print("DEBUG: splitting both X and Y in train and test data")
        # TODO: model_selection.train_test_split() function name is misleading,
        # there is the same function in sklearn.model_selection
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X_cs,
            Y,
            test_size=1 - self.config.getfloat("machine_learning", "train_fraction"),
        )

        # for training-set and test-set, split in ID, geometry, and values
        X_train_ID, X_train_geom, X_train = conflict.split_conflict_geom_data(X_train)
        X_test_ID, X_test_geom, X_test = conflict.split_conflict_geom_data(X_test)

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fits classifier based on training-data and makes predictions.
        The fitted classifier is dumped to file with pickle to be used again during projections.
        Makes prediction with test-data including probabilities of those predictions.

        Args:
            X_train (np.ndarray, pd.DataFrame): training-data of variable values.
            y_train (np.ndarray): training-data of conflict data.
            X_test (np.ndarray, pd.DataFrame): test-data of variable values.
            out_dir (str): path to output folder.
            run_nr (int): number of fit/predict repetition and created classifier.

        Returns:
            arrays: arrays including the predictions made and their probabilities
        """

        # fit the classifier with training data
        self.clf.fit(X_train, y_train)

        # create folder to store all classifiers with pickle
        clf_pickle_rep = os.path.join(out_dir, "clfs")
        if not os.path.isdir(clf_pickle_rep):
            os.makedirs(clf_pickle_rep)

        # save the fitted classifier to file via pickle.dump()
        if self.config.getboolean("general", "verbose"):
            print("DEBUG: dumping classifier to {}".format(clf_pickle_rep))
        with open(os.path.join(clf_pickle_rep, "clf_{}.pkl".format(run_nr)), "wb") as f:
            pickle.dump(self.clf, f)

        # make prediction
        y_pred = self.clf.predict(X_test)
        # make prediction of probability
        y_prob = self.clf.predict_proba(X_test)

        return y_pred, y_prob


def load_clfs(config: RawConfigParser, out_dir: str) -> list[str]:
    """Loads the paths to all previously fitted classifiers to a list.
    Classifiers were saved to file in fit_predict().
    With this list, the classifiers can be loaded again during projections.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        out_dir (path): path to output folder.

    Returns:
        list: list with file names of classifiers.
    """

    clfs = os.listdir(os.path.join(out_dir, "clfs"))

    assert len(clfs) == config.getint("machine_learning", "n_runs"), AssertionError(
        "ERROR: number of loaded classifiers does not match the specified number of runs in cfg-file!"
    )

    return clfs


def _define_scaling(
    config: RawConfigParser,
) -> Union[
    preprocessing.MinMaxScaler,
    preprocessing.StandardScaler,
    preprocessing.RobustScaler,
    preprocessing.QuantileTransformer,
]:
    """Defines scaling method based on model configurations.

    Args:
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.

    Raises:
        ValueError: raised if a non-supported scaling method is specified.

    Returns:
        scaler: the specified scaling method instance.
    """

    if config.get("machine_learning", "scaler") == "MinMaxScaler":
        scaler = preprocessing.MinMaxScaler()
    elif config.get("machine_learning", "scaler") == "StandardScaler":
        scaler = preprocessing.StandardScaler()
    elif config.get("machine_learning", "scaler") == "RobustScaler":
        scaler = preprocessing.RobustScaler()
    elif config.get("machine_learning", "scaler") == "QuantileTransformer":
        scaler = preprocessing.QuantileTransformer(random_state=42)
    else:
        raise ValueError(
            "no supported scaling-algorithm selected - \
                choose between MinMaxScaler, StandardScaler, RobustScaler or QuantileTransformer"
        )

    if config.getboolean("general", "verbose"):
        print("DEBUG: chosen scaling method is {}".format(scaler))

    return scaler
