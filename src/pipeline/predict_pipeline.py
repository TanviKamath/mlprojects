import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            print("[DEBUG] Starting prediction...")
            model_path = os.path.join('artifacts', 'model.pkl') 
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            print("[DEBUG] Loading model from:", model_path)
            model = load_object(model_path)
            
            print("[DEBUG] Loading preprocessor from:", preprocessor_path)
            preprocessor = load_object(preprocessor_path)

            print("[DEBUG] Transforming input features...")
            data_scaled = preprocessor.transform(features)

            print("[DEBUG] Making prediction...")
            predictions = model.predict(data_scaled)

            print("[DEBUG] Predictions:", predictions)
            return predictions
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise CustomException(f"Error in prediction pipeline: {str(e)}")

class CustomData:
    def __init__(self, 
        gender:str,
        race_ethnicity:str,
        parental_level_of_education:str,
        lunch:str,
        test_preparation_course:str,
        reading_score:int,
        writing_score:int
        ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": self.gender,
                "race_ethnicity": self.race_ethnicity,
                "parental_level_of_education": self.parental_level_of_education,
                "lunch": self.lunch,
                "test_preparation_course": self.test_preparation_course,
                "reading_score": self.reading_score,
                "writing_score": self.writing_score
            }
            return pd.DataFrame([custom_data_input_dict])
        except Exception as e:
            raise CustomException(f"Error in converting data to DataFrame: {str(e)}")