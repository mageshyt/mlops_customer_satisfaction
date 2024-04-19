import logging
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract class for data cleaning.
    """

    @abstractmethod
    def __init__(self, data: Union[pd.DataFrame, str]):
        """
        Initialize the data cleaning object.

        :param data: The data to clean.
        """
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            raise ValueError("data must be a pandas DataFrame or a path to a CSV file.")

    @abstractmethod
    def handle_data(self) ->Union[pd.DataFrame,pd.Series]:
        """
        Clean the data.

        :return: The cleaned data.
        """
        pass



class DataPreProcessStrategy(DataStrategy):
    """
    Class for data cleaning using preprocessing techniques.
    """
    def __init__(self, data: Union[pd.DataFrame, str]):
        """
        Initialize the data cleaning object.

        :param data: The data to clean.
        """
        super().__init__(data)

    def handle_data(self) -> pd.DataFrame:
        """
        Clean the data using preprocessing techniques.

        :return: The cleaned data.
        """
        try :
            logging.info("Cleaning the data using preprocessing techniques.")
            self.data = self.data.drop([
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp",
            ], axis=1) # Drop unnecessary columns


            return self.data
        except Exception as e:
            logging.error(f"An error occurred while cleaning the data: {e}")
            raise e
    
    