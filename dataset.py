import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from typing import Dict, Generator, List, Tuple, Union

class VorAusDataset:
    def __init__(self, path: str):
        """
        Read the dataset passed by input and proceed to remove all the columns not used for the project.
        Next it converts the column anomaly from boolean to int.
        """
        if path is None:
            raise ValueError('Path variable is empty')
        self.dataframe = pd.read_parquet(path)
        self.dataframe = self.dataframe.drop(['time','sample','category','setting','action','active'], axis=1)
        self.dataframe['anomaly'] = self.dataframe['anomaly'].astype(int)

    def get_prepared_dataset_tts(self,test_size: int, random_state:int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Returns the dataset splitted in X_train, y_train, X_test, y_test by using the train_test_split procedure
        of the module sklearn.
        Args:
            test_size: indicates the % of the splitting.
            random_state: define the random state for the reproducibily of the procedure
        Returns:
            X_train: is the data which is used to train
            y_train: the labels of the data to train
            X_test: is the data which is used for testing the procedure of the training
            y_test: real values used for calculating the performances of the model.
        """
        if self.dataframe is not None:
            y = self.dataframe['anomaly'].to_numpy()
            X = self.dataframe.drop('anomaly',axis=1).to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            return X_train, y_train, X_test, y_test
        raise Exception('Dataset not initialized!')

    def get_prepared_dataset_ad(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Returns the dataset splitted in X_train, y_train, X_test, y_test in which the X_train only contains 'clean values'
        so rows for which there has not been an anomaly and the test set includes all the observation for which an anomaly
        has been detected.
        Returns:
            X_train: is the data which is used to train
            y_train: the labels of the data to train
            X_test: is the data which is used for testing the procedure of the training
            y_test: real values used for calculating the performances of the model.
        """
        clean_record = self.dataframe.anomaly == 0

        X_train = self.dataframe[clean_record].to_numpy()
        X_test = self.dataframe[~clean_record].to_numpy()
        y_train = self.dataframe[clean_record].anomaly.to_numpy()
        y_test = self.dataframe[~clean_record].anomaly.to_numpy()

        return X_train, y_train, X_test, y_test
