import sys
import pandas as pd
from src.exception import ExceptionHandler
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame) -> pd.Series:
        try:
            logging.info("Loading the preprocessor and model objects")
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model_path = 'artifacts/model.pkl'

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            logging.info("Transforming the input features")
            data_scaled = preprocessor.transform(features)

            logging.info("Making predictions")
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            raise ExceptionHandler(e, sys)

class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        reading_score: float,
        lunch: str,
        parental_level_of_education: str,
        writing_score: float,
        test_preparation_course: str
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.reading_score = reading_score
        self.lunch = lunch
        self.parental_level_of_education = parental_level_of_education
        self.writing_score = writing_score
        self.test_preparation_course = test_preparation_course

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "reading_score": [self.reading_score],
                "lunch": [self.lunch],
                "parental_level_of_education": [self.parental_level_of_education],
                "writing_score": [self.writing_score],
                "test_preparation_course": [self.test_preparation_course],
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise ExceptionHandler(e, sys)