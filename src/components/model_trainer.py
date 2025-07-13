import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting model training process")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "SVR": SVR(),
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0),
                "XGBRegressor": XGBRegressor(eval_metric='rmse')
            }
            params = {
                "KNeighborsRegressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"]
                },
                "DecisionTreeRegressor": {
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },
                "RandomForestRegressor": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10, 20]
                },
                "AdaBoostRegressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0]
                },
                "SVR": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"]
                },
                "LinearRegression": {},
                "Ridge": {
                    "alpha": [0.1, 1.0, 10.0]
                },
                "Lasso": {
                    "alpha": [0.1, 1.0, 10.0]
                },
                "CatBoostRegressor": {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [100, 200]
                },
                "XGBRegressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 6, 10]
                }
            }


            logging.info("Evaluating models")
            model_report :dict=evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_name = max(model_report, key=lambda k: model_report[k]['test_score'])
            best_model_score = model_report[best_model_name]['test_score']

            logging.info(f"Best model found: {best_model_name}")

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)

            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}")


            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)