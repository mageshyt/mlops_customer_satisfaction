import logging
from zenml.steps import BaseParameters
from zenml import  step

import pandas as pd
from sklearn.base import RegressorMixin
from model.model_devlopment import LinearRegression,RandomForest

from  .config import ModelNameconfig
@step
def model_train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config:ModelNameconfig
) -> RegressorMixin:
    """
    Training the model.

    Args:
        X_train (pd.DataFrame): Training data.
        X_test (pd.DataFrame): Testing data.
        y_train (pd.DataFrame): Training labels.
        y_test (pd.DataFrame): Testing labels.
    """
    logging.info("Training the model")
    # Training the model
    model=None

    if config.model_type=="LinearRegression":
        model=LinearRegression()
        trained_mode=model.train(X_train,y_train)

        return trained_mode

    elif config.model_type=="RandomForest":
        model=RandomForest()
        trained_mode=model.train(X_train,y_train)
        return trained_mode
    else:
        raise ValueError(f"Invalid model type: {config.model_type}")
