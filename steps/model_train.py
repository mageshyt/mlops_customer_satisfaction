import logging
from zenml.steps import BaseParameters
from zenml import  step

import pandas as pd
from sklearn.base import RegressorMixin
from model.model_devlopment import LinearRegressionModel,RandomForestModel

from  .config import ModelNameConfig
@step
def model_train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config:ModelNameConfig
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

    if config.model_name=="LinearRegression":
        model=LinearRegressionModel()
        trained_mode=model.train(X_train,y_train)

        return trained_mode

    elif config.model_name=="RandomForest":
        model=RandomForestModel()
        trained_mode=model.train(X_train,y_train)
        return trained_mode
    else:
        raise ValueError(f"Invalid model type: {config.model_name}")
