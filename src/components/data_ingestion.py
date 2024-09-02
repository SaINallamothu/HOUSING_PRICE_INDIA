import os
import sys

import pandas as pd

from src.logger import logging
from src.exceptions import CustomException
from src.utils import load_csv

from src.components.data_transformation import DatatransformationConfig, DataTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path: str= os.path.join("artifacts/CSV_files", "raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Combining all csv's from base_files Dir to a raw_data.csv")
        try:
            combined_df=  load_csv('artifacts/base_files')   #----------> Here enter a base_files Dir path
            print(combined_df.head())

            os.makedirs("artifacts/CSV_files", exist_ok=True) #creating a CSV_files directory

            combined_df.to_csv(self.ingestion_config.raw_data_path) #saving combined dataframe ton raw_data.csv 
            return combined_df

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    data= DataIngestion()
    df=data.initiate_data_ingestion()

    data_transformation= DataTransformation()
    data_transformation.DataDropping(df)
    data_transformation.Datasplit()
    train_arr, test_arr,save_obj=data_transformation.Initiate_data_transformation()

    model= ModelTrainer()
    model.initiate_model_training(train_arr, test_arr)

