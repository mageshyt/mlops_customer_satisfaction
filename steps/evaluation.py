import logging
from zenml import step

import pandas as pd
from sklearn.base import RegressorMixin
from model.model_evaluation import MAE, RMSE, R2,MSE
import mlflow

from zenml.client import Client

from typing import Annotated, Tuple
experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model:RegressorMixin,
                X_test: pd.DataFrame,
                y_test: pd.Series,
             ) -> Tuple[
                Annotated[float, 'r2'],
                Annotated[float, 'mae'],
                Annotated[float, 'rmse'],
                Annotated[float, 'mse']
             ]:
    """
    Evaluating the model.

    Args:
        data (pd.DataFrame): data to evaluate the model
    """
    try:
        logging.info("Evaluating the model")
        # Making predictions
        predictions=model.predict(X_test)
        # Evaluating the model
        mae=MAE().evaluate(y_test,predictions)
        rmse=RMSE().evaluate(y_test,predictions)
        r2=R2().evaluate(y_test,predictions)
        mse=MSE().evaluate(y_test,predictions)

        logging.info(f"Mean Absolute Error: {mae}")
        logging.info(f"Root Mean Squared Error: {rmse}")
        logging.info(f"R2: {r2}")

        mlflow.log_metric("mae",mae)
        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric("r2",r2)
        mlflow.log_metric("mse",mse)
        

        return r2,mae,rmse,mse
    
    except Exception as e:
        logging.error(f"An error occurred while evaluating the model: {e}")
        raise e
