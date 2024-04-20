import logging
import pandas as pd
from zenml import step

from model.model_devlopment import LinearRegression,RandomForest
@step
def model_train(data: pd.DataFrame) -> None:
    """
    Training the model.

    Args:
        data (pd.DataFrame): data to train the model
    """
    logging.info("Training the model")
    # Training the model
    pass

