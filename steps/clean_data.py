import logging
import pandas as pd
from zenml import step


@step
def clean_data(data: pd.DataFrame) -> None:
    """
    Cleaning the data.

    Args:
        data (pd.DataFrame): data to be cleaned

    Returns:
        pd.DataFrame: cleaned data
    """
    logging.info("Cleaning data")
    # Dropping rows with missing values
    
    # return data.dropna()