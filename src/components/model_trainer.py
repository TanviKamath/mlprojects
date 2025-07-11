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

            logging.info("Evaluating models")
            model_report = evaluate_models(X_train, y_train, models)

            best_model_name = max(model_report, key=lambda name: model_report[name]['test_score'])
            best_model_score = model_report[best_model_name]['test_score']
            logging.info(f"Best model found: {best_model_name}")

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model_name
            )
            logging.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}")


            predicted = models[best_model_name].predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)