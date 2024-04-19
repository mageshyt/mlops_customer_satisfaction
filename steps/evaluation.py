import logging
from zenml import step
import pandas as pd

@step
def evaluate(data: pd.DataFrame) -> None:
    """
    Evaluating the model.

    Args:
        data (pd.DataFrame): data to evaluate the model
    """
    logging.info("Evaluating the model")
    # Evaluating the model
    pass