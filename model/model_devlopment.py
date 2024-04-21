import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import optuna
import pandas as pd
import xgboost as xgb

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
    
    @abstractmethod
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        """
        Optimizes the hyperparameters of the model.

        Args:
            trial: Optuna trial object
            x_train: Training data
            y_train: Target data
            x_test: Testing data
            y_test: Testing target
        """
        pass




class LinearRegressionModel(Model):
    """
    Class for linear regression model development.
    """

    def train(self,x_train,y_train,**kwargs):
        try:
            logging.info("Linear regression model developed.")
            reg=LinearRegression(**kwargs)
            reg.fit(x_train,y_train)
            return reg
        except Exception as e:
            logging.error(f"An error occurred while developing the linear regression model: {e}")
            raise e
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        reg=self.train(x_train,y_train)
        return reg.score(x_test,y_test)
    

class RandomForestModel(Model):
    """
    Class for Random Forest model development.
    """

    def train(self,x_train,y_train,**kwargs):
        try:
            logging.info("Random Forest model developed.")
            reg=RandomForestRegressor(**kwargs)
            reg.fit(x_train,y_train)
            return reg
        except Exception as e:
            logging.error(f"An error occurred while developing the Random Forest model: {e}")
            raise e
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        reg = self.train(x_train, y_train, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
        return reg.score(x_test, y_test)
    


class XGBoostModel(Model):
    """
    XGBoostModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        reg = xgb.XGBRegressor(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 30)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 10.0)
        reg = self.train(x_train, y_train, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        return reg.score(x_test, y_test)


class HyperparameterTuner:
    """
    Class for performing hyperparameter tuning. It uses Model strategy to perform tuning.
    """

    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.model.optimize(trial, self.x_train, self.y_train, self.x_test, self.y_test), n_trials=n_trials)
        return study.best_trial.params

