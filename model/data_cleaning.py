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

        data: The data to clean.
        """
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            raise ValueError("data must be a pandas DataFrame or a path to a CSV file.")

    @abstractmethod
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
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
        try:
            logging.info("Cleaning the data using preprocessing techniques.")

            self.data = self.data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                    "order_status",
                    "customer_zip_code_prefix",
                    "order_item_id",
                ],
                axis=1,
            )  # Drop unnecessary columns

            # fill up the null | missing values  (refer to the EDA)

            self.data["product_weight_g"].fillna(
                self.data["product_weight_g"].mean(), inplace=True
            )
            self.data["product_length_cm"].fillna(
                self.data["product_length_cm"].mean(), inplace=True
            )
            self.data["product_height_cm"].fillna(
                self.data["product_height_cm"].mean(), inplace=True
            )
            self.data["product_width_cm"].fillna(
                self.data["product_width_cm"].mean(), inplace=True
            )
            self.data["review_comment_message"].fillna("No comment", inplace=True)

            # pick the np.number data type columns
            self.data = self.data.select_dtypes(include=[np.number])

            return self.data
        except Exception as e:
            logging.error(f"An error occurred while cleaning the data: {e}")
            raise e


class DataSplitStrategy(DataStrategy):
    """
    Divide the data into training and testing sets.
    """

    def __init__(self, data: Union[pd.DataFrame, str]):
        """
        Initialize the data cleaning object.

        :param data: The data to clean.
        """
        super().__init__(data)

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Clean the data using splitting techniques.

        :return: The cleaned data.
        """
        try:
            logging.info("Cleaning the data using splitting techniques.")

            # Split the data into features and target
            X = self.data.drop("review_score", axis=1)  # Features
            y = self.data["review_score"]  # Target variable

            # Split the data into training and testing sets (80% training, 20% testing)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"An error occurred while cleaning the data: {e}")
            raise e


class DataCleaning:
    """
    Class for clearning data using a strategy then divide the data into training and testing sets.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        """
        Initialize the data cleaner object.

        data: The data to clean.
        strategy: The strategy to use for cleaning the data.
        """
        self.data = data
        self.strategy = strategy

    def clean_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Clean the data using the specified strategy.

        :return: The cleaned data.
        """
        try:
            return self.strategy.handle_data()
        except Exception as e:
            logging.error(f"An error occurred while cleaning the data: {e}")
            raise e


if __name__ == "__main__":
    data = pd.read_csv(
        "/Volumes/Project-2/programming/machine_deep_learning/projects/customer_satisfaction/data/olist_customers_dataset.csv"
    )
    data_cleaner = DataCleaning(data, DataPreProcessStrategy(data))
    cleaned_data = data_cleaner.clean_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = DataSplitStrategy(cleaned_data).handle_data()

    print("Data cleaning and splitting complete.")

    print("Size of training set:", X_train.shape)
    print("Size of testing set:", X_test.shape)
    print("Size of training labels:", y_train.shape)
    print("Size of testing labels:", y_test.shape)
