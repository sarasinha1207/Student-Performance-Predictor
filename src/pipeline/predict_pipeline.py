import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("models", "model.joblib")
            preprocessor_path = os.path.join("models", "preprocessor.joblib")
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        previous_score: float,
        attendance: float,
        assignments_completed: int,
        study_hours: float,
        class_participation: int,
        failures: int,
        sleep_hours: float,
        screen_time: float,
        physical_activity: int,
        travel_time: str,
        internet_access: str,
        school_type: str
    ):
        self.previous_score = previous_score
        self.attendance = attendance
        self.assignments_completed = assignments_completed
        self.study_hours = study_hours
        self.class_participation = class_participation
        self.failures = failures
        self.sleep_hours = sleep_hours
        self.screen_time = screen_time
        self.physical_activity = physical_activity
        self.travel_time = travel_time
        self.internet_access = internet_access
        self.school_type = school_type

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "previous_score": [self.previous_score],
                "attendance": [self.attendance],
                "assignments_completed": [self.assignments_completed],
                "study_hours": [self.study_hours],
                "class_participation": [self.class_participation],
                "failures": [self.failures],
                "sleep_hours": [self.sleep_hours],
                "screen_time": [self.screen_time],
                "physical_activity": [self.physical_activity],
                "travel_time": [self.travel_time],
                "internet_access": [self.internet_access],
                "school_type": [self.school_type]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
