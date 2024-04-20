import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
class Evaluation(ABC):
    """
    Abstract class for model evaluation.
    """

    @abstractmethod
    def evaluate(self,  y_true:np.ndarray, y_pred:np.ndarray):
        """
        Evaluate the model.

        Args:
            model: The trained model.
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        """
        pass


class MSE(Evaluation):
    """
    Class for Mean Squared Error evaluation.
    """

    def evaluate(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        Evaluate the model using Mean Squared Error.

        Args:
            model: The trained model.
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        """
        try:
            logging.info("Evaluating the model using Mean Squared Error.")
            mse = mean_squared_error(y_true, y_pred)

            return mse
        except Exception as e:
            logging.error(f"An error occurred while evaluating the model using Mean Squared Error: {e}")
            raise e
        

class MAE(Evaluation):
    """
    Class for Mean Absolute Error evaluation.
    """

    def evaluate(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        Evaluate the model using Mean Absolute Error.

        Args:
            model: The trained model.
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        """
        try:
            logging.info("Evaluating the model using Mean Absolute Error.")
            mae = mean_absolute_error(y_true, y_pred)

            return mae
        except Exception as e:
            logging.error(f"An error occurred while evaluating the model using Mean Absolute Error: {e}")
            raise e
        

class R2(Evaluation):
    """
    Class for R2 evaluation.
    """

    def evaluate(self,y_true:np.ndarray, y_pred:np.ndarray):
        """
        Evaluate the model using R2.

        Args:
            model: The trained model.
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        """
        try:
            logging.info("Evaluating the model using R2.")
            r2 = r2_score(y_true, y_pred)

            return r2
        except Exception as e:
            logging.error(f"An error occurred while evaluating the model using R2: {e}")
            raise e
        
class RMSE(Evaluation):
    """
    Class for Root Mean Squared Error evaluation.
    """

    def evaluate(self,  y_true:np.ndarray, y_pred:np.ndarray):
        """
        Evaluate the model using Root Mean Squared Error.

        Args:
            model: The trained model.
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        """
        try:
            logging.info("Evaluating the model using Root Mean Squared Error.")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            return rmse
        except Exception as e:
            logging.error(f"An error occurred while evaluating the model using Root Mean Squared Error: {e}")
            raise e