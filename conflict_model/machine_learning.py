import pandas as pd
import seaborn as sbs
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import os, sys

def prepare_data(df, Xvars, Yvar):

    if len(Xvars) < 2:
        raise ValueError('at least 2 variables need to be specified!')
    if len(yvar) > 1:
        raise ValueError('maximum 1 target variable must be specified!')

    Y  = np.append(Y, df[yvar].values)

    return X, y
