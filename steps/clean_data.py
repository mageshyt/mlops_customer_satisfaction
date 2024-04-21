import logging
import pandas as pd
from zenml import step
from model.data_cleaning import DataCleaning, DataPreProcessStrategy, DataSplitStrategy

from typing import Annotated, Tuple


@step
def clean_data(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Cleaning the data.

    Args:
        data (pd.DataFrame): data to be cleaned

    Returns:
         Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: cleaned data
    """
    try:
        logging.info("Cleaning data")
        # Data cleaning
        process_strategy = DataPreProcessStrategy(data)
        data_cleaning = DataCleaning(data, process_strategy)
        processed_data = data_cleaning.clean_data()
        # Splitting the data
        split_strategy = DataSplitStrategy(processed_data)

        X_train, X_test, y_train, y_test = split_strategy.handle_data()

        logging.info("Data cleaning complete !")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"An error occurred while cleaning the data: {e}")
        raise e
