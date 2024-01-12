from copro import machine_learning, conflict, evaluation
import pandas as pd


def all_data(X, Y, config, scaler, clf, out_dir, run_nr):
    """Main model workflow when all XY-data is used.
    The model workflow is executed for each classifier.

    Args:
        X (array): array containing the variable values plus IDs and geometry information.
        Y (array): array containing merely the binary conflict classifier data.
        config (ConfigParser-object): object containing the parsed configuration-settings of the model.
        scaler (scaler): the specified scaling method instance.
        clf (classifier): the specified model instance.
        out_dir (str): path to output folder.

    Returns:
        dataframe: containing the test-data X-array values.
        datatrame: containing model output on polygon-basis.
        dict: dictionary containing evaluation metrics per simulation.
    """
    if config.getboolean("general", "verbose"):
        print("DEBUG: using all data")

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
    ) = machine_learning.split_scale_train_test_split(X, Y, config, scaler)

    # convert to dataframe
    X_df = pd.DataFrame(X_test)

    # fit classifier and make prediction with test-set
    y_pred, y_prob = machine_learning.fit_predict(
        X_train, y_train, X_test, clf, config, out_dir, run_nr
    )
    y_prob_0 = y_prob[:, 0]  # probability to predict 0
    y_prob_1 = y_prob[:, 1]  # probability to predict 1

    # evaluate prediction and save to dict
    eval_dict = evaluation.evaluate_prediction(
        y_test, y_pred, y_prob, X_test, clf, config
    )

    # aggregate predictions per polygon
    y_df = conflict.get_pred_conflict_geometry(
        X_test_ID, X_test_geom, y_test, y_pred, y_prob_0, y_prob_1
    )

    return X_df, y_df, eval_dict
