import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_path = os.path.join("artifacts","train.csv")
    test_path = os.path.join("artifacts","test.csv")
    raw_path = os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Entered into data ingestion")
        try:
            df = pd.read_csv("notebook/data/brain_stroke.csv")
            logging.info("Reads train and test data in data ingestion part")

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_path),exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_path,header=True,index=False)
            logging.info("Train Test split is initiated")

            train_set, test_set = train_test_split(df,test_size=0.25,random_state=42)

            train_set.to_csv(self.data_ingestion_config.train_path,header=True,index=False)
            test_set.to_csv(self.data_ingestion_config.test_path,header=True,index=False)

            logging.info("Data ingestion is completed")

            return (
                self.data_ingestion_config.train_path,
                self.data_ingestion_config.test_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    data_ingestion = DataIngestion()
    train_data,test_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr,test_arr)