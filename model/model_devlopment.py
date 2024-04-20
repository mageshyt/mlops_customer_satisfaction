import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class Model(ABC):
    """
    Abstract class for model development.
    """

    @abstractmethod
    def train(self,x_train,y_train):
        """
        Develop the model.

        Args:
            x_train (pd.series):  training data 
            y_train (pd.serires):  labels

        Returns:
            model: The developed model.
        """
        pass



class LinearRegressionModel(Model):
    """
    Class for linear regression model development.
    """

    def train(self,x_train,y_train,**kwargs):
        """
        Develop the linear regression model.

        Args:
            x_train (pd.series):  training data 
            y_train (pd.serires):  labels

        Returns:
            model: The developed model.
        """
        try:
            logging.info("Linear regression model developed.")
            reg=LinearRegression(**kwargs)
            reg.fit(x_train,y_train)
            return reg
        except Exception as e:
            logging.error(f"An error occurred while developing the linear regression model: {e}")
            raise e
        


class RandomForestModel(Model):
    """
    Class for Random Forest model development.
    """

    def train(self,x_train,y_train,**kwargs):
        """
        Develop the Random Forest model.

        Args:
            x_train (pd.series):  training data 
            y_train (pd.serires):  labels

        Returns:
            model: The developed model.
        """
        try:
            logging.info("Random Forest model developed.")
            reg=RandomForestRegressor(**kwargs)
            reg.fit(x_train,y_train)
            return reg
        except Exception as e:
            logging.error(f"An error occurred while developing the Random Forest model: {e}")
            raise e
        
        


