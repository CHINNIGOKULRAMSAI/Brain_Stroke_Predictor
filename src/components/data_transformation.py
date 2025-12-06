import os 
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def preprocessor_func(self):
        logging.info("Entered into data transformation part")
        try:
            num_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
            cat_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('std_scaler',StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoder',OneHotEncoder(handle_unknown='ignore')),
                    ('std_scaler',StandardScaler(with_mean=False)),
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,num_features),
                ('cat_pipeline',cat_pipeline,cat_features),
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor = self.preprocessor_func()

            target_column = ['stroke']

            input_train_fea = train_df.drop(columns=target_column,axis=1)
            target_train_fea = train_df[target_column]

            input_test_fea = test_df.drop(columns=target_column,axis=1)
            target_test_fea = test_df[target_column]

            logging.info("Preprocessing is initiated")

            input_train_fea = preprocessor.fit_transform(input_train_fea)
            input_test_fea = preprocessor.transform(input_test_fea)

            train_arr = np.c_[input_train_fea,np.array(target_train_fea)]
            test_arr = np.c_[input_test_fea,np.array(target_test_fea)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_path,
                obj = preprocessor,
            )

            logging.info("Data transformation is completed")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )

        except Exception as e:
            raise CustomException(e,sys)