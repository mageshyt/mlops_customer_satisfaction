import logging


from zenml import  step
from zenml.client import Client
import mlflow

import pandas as pd
from sklearn.base import RegressorMixin
from model.model_devlopment import LinearRegressionModel,RandomForestModel,XGBoostModel,HyperparameterTuner,AdaBoostModel

from  .config import ModelNameConfig


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def model_train(
       x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Training the model.

    Args:
        X_train (pd.DataFrame): Training data.
        X_test (pd.DataFrame): Testing data.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
    """
    try:
        logging.info("Training the model")
        # Training the model
        model=None
        tuner=None

        if config.model_name == "randomforest":
                mlflow.sklearn.autolog()
                model = RandomForestModel()
        elif config.model_name == "xgboost":
                mlflow.xgboost.autolog()
                model = XGBoostModel()
        elif config.model_name == "linear_regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
        elif config.model_name == "adaboost":
            mlflow.sklearn.autolog()
            model = AdaBoostModel()
            
        else:
                raise ValueError(f"Invalid model type: {config.model_name}")
            
        tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)
        
        if config.fine_tuning:
            best_model = tuner.optimize()
            trained_model = model.train(x_train, y_train, **best_model)
        else:
            trained_model = model.train(x_train, y_train)
            
        return trained_model
    except Exception as e:
        logging.error(f"An error occurred while training the model: {e}")
        raise e

