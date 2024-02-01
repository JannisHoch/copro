from copro import machine_learning, conflict, evaluation
from configparser import RawConfigParser
from sklearn import preprocessing, ensemble
import pandas as pd
import numpy as np
from typing import Union


class MainModel:
    def __init__(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        Y: np.ndarray,
        config: RawConfigParser,
        scaler: Union[
            preprocessing.MinMaxScaler,
            preprocessing.StandardScaler,
            preprocessing.RobustScaler,
            preprocessing.QuantileTransformer,
        ],
        clf: ensemble.RandomForestClassifier,
        out_dir: str,
        run_nr: int,
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
        self.scaler = scaler
        self.clf = clf
        self.out_dir = out_dir
        self.run_nr = run_nr

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Main model workflow when all XY-data is used.
        The model workflow is executed for each classifier.

        Returns:
            dataframe: containing the test-data X-array values.
            datatrame: containing model output on polygon-basis.
            dict: dictionary containing evaluation metrics per simulation.
        """
        if self.config.getboolean("general", "verbose"):
            print("DEBUG: using all data")

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
            X_train, y_train, X_test, self.out_dir, self.run_nr
        )
        y_prob_0 = y_prob[:, 0]  # probability to predict 0
        y_prob_1 = y_prob[:, 1]  # probability to predict 1

        # evaluate prediction and save to dict
        eval_dict = evaluation.evaluate_prediction(
            y_test, y_pred, y_prob, X_test, self.clf, self.config
        )

        # aggregate predictions per polygon
        y_df = conflict.get_pred_conflict_geometry(
            X_test_ID, X_test_geom, y_test, y_pred, y_prob_0, y_prob_1
        )

        return X_df, y_df, eval_dict
