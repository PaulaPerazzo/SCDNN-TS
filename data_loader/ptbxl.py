import numpy as np
import pandas as pd
from typing import Tuple

from data_loader.base import DataLoader


class PTBXL(DataLoader):
    def load(self) -> Tuple[np.ndarray, pd.DataFrame]:
        ### -------------------------------
        # load data and target
        print("Data Loading start!")
        print("---------------------------")

        path = "./data/ptbxl_database/"

        # # for benchmark
        # P_X_train, P_X_test, P_X_val = (
        #     path + "x_train.npy",
        #     path + "x_test.npy",
        #     path + "x_val.npy",
        # )
        # P_y_train, P_y_test, P_y_val = (
        #     path + "y_train.npy",
        #     path + "y_test.npy",
        #     path + "y_val.npy",
        # )
        # ###

        # X_train = np.load(P_X_train, allow_pickle=True)
        # X_test = np.load(P_X_test, allow_pickle=True)
        # X_val = np.load(P_X_val, allow_pickle=True)

        # print("X_train loaded:", X_train.shape)

        # # for benchmark
        # X_train = np.swapaxes(X_train, 1, 2)
        # X_test = np.swapaxes(X_test, 1, 2)
        # X_val = np.swapaxes(X_val, 1, 2)
        # ###

        # y_train = np.load(P_y_train, allow_pickle=True)
        # y_test = np.load(P_y_test, allow_pickle=True)
        # y_val = np.load(P_y_val, allow_pickle=True)

        # # for benchmark
        # y_train = np.argmax(y_train, axis=1)
        # y_test = np.argmax(y_test, axis=1)
        # y_val = np.argmax(y_val, axis=1)

        X = np.load(path + "x.npy", allow_pickle=True)
        X = np.swapaxes(X, 1, 2)
        y = pd.read_csv(path + "y.csv")

        return X, y
