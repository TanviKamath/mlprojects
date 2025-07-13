import os
import sys

import pandas as pd
import numpy as np
import dill
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV


from src.exception import CustomException

def save_object(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(x_train, y_train,x_test, y_test, models, params):
    try:
    
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            params_model = params[list(models.keys())[i]]
            rs = RandomizedSearchCV(model, param_distributions=params_model, cv=3)
            rs.fit(x_train, y_train)

            model.set_params(**rs.best_params_)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_train_pred = model.predict(x_train)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_pred)
            report[list(models.keys())[i]] = {"train_score": train_model_score, "test_score": test_model_score}
        return report

    except Exception as e:
            raise CustomException(f"Error occurred while evaluating models: {e}", sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(f"Error occurred while loading object from {file_path}: {e}", sys)