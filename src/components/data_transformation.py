import math
import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Creating data transformation pipeline")

            # Define the numerical and categorical columns
            numerical_cols = ["writing_score", "reading_score"]
            categorical_cols = ["gender","race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            # Create the numerical transformation pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            # Create the categorical transformation pipeline
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine the numerical and categorical pipelines
            preprocessor = ColumnTransformer(transformers=[
                ('num', num_pipeline, numerical_cols),
                ('cat', cat_pipeline, categorical_cols)
            ])

            logging.info("Data transformation pipelines created successfully")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            preprocessing_obj = self.get_data_transformer_object()
            target_column = "math_score"  
            numerical_cols = ["writing_score", "reading_score"]
            input_features_train = train_df.drop(columns=[target_column], axis=1)   
            input_features_test = test_df.drop(columns=[target_column], axis=1)
            target_feature_train = train_df[target_column]
            target_feature_test = test_df[target_column]
            logging.info("Starting data transformation")
            input_train = preprocessing_obj.fit_transform(input_features_train)
            input_test = preprocessing_obj.transform(input_features_test)

            train_arr = np.c_[input_train, np.array(target_feature_train)]
            test_arr = np.c_[input_test, np.array(target_feature_test)]

            logging.info("Data transformation completed successfully")

            # Save the preprocessor object
            save_object(obj=preprocessing_obj, file_path=self.data_transformation_config.preprocessor_path)

            return train_arr, test_arr, self.data_transformation_config.preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)
        