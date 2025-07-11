import os
import sys

import pandas as pd
import numpy as np
import dill
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


from src.exception import CustomException

def save_object(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(x,y, models):
    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)  
            train_model_score = r2_score(y_train, model.predict(x_train))
            test_model_score = r2_score(y_test, y_pred)
            report[list(models.keys())[i]] = {"train_score": train_model_score, "test_score": test_model_score}
        return report

    except Exception as e:
            raise CustomException(f"Error occurred while evaluating models: {e}", sys)