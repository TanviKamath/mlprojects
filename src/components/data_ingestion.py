import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass



# Automatically add the project root (C:\mlprojects) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    dataset_path: str = os.path.join("notebook", "stud.csv")
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    split_ratio: float = 0.2

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion process")

            # Read dataset
            df = pd.read_csv(self.config.dataset_path)
            logging.info("Dataset loaded successfully")

            # Ensure artifacts directory exists
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.config.raw_data_path, index=False)
            logging.info(f"Raw data saved at {self.config.raw_data_path}")

            # Split into train and test
            train_df, test_df = train_test_split(df, test_size=self.config.split_ratio)
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logging.info(f"Train data saved at {self.config.train_data_path}")
            logging.info(f"Test data saved at {self.config.test_data_path}")
            logging.info("Data ingestion completed successfully")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_path, test_path)
    print(f"Train data saved at: {train_path}")
    print(f"Test data saved at: {test_path}")
