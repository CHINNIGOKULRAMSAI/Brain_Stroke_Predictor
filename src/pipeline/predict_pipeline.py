import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData:
    def __init__(self,gender, age, hypertension, heart_disease, ever_married,
       work_type, Residence_type, avg_glucose_level, bmi,
       smoking_status):
        self.gender = gender
        self.age = age
        self.hypertension = hypertension
        self.heart_disease = heart_disease
        self.ever_married = ever_married
        self.work_type = work_type
        self.Residence_type = Residence_type
        self.avg_glucose_level = avg_glucose_level
        self.bmi = bmi
        self.smoking_status = smoking_status
    def get_data_as_dataFrame(self):
        try:
            data = {
                'gender' : [self.gender],
                'age' : [float(self.age)],
                'hypertension' : [int(self.hypertension)],
                'heart_disease' : [int(self.heart_disease)],
                'ever_married' : [self.ever_married],
                'work_type' : [self.work_type],
                'Residence_type' : [self.Residence_type],
                'avg_glucose_level' : [float(self.avg_glucose_level)],
                'bmi' : [float(self.bmi)],
                'smoking_status' : [self.smoking_status],
            }

            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e,sys)