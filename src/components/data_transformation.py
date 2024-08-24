import sys
import os
import pandas as pd

from src.logger import logging
from src.exceptions import CustomException
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DatatransformationConfig:
    raw_data01_path: str=os.path.join("artifacts/CSV_files", "raw_data01.csv")
    train_data_path: str= os.path.join("artifacts/CSV_files", "train_data.csv")
    test_data_path: str= os.path.join("artifacts/CSV_files", "test_data.csv")

class DataTransformation:
    def __init__(self):
        self.transformation_config=DatatransformationConfig()

    def DataDropping(self, dataframe):
        try:
            org_shape = dataframe.shape
            print("Shape of df after 10 columns removed: {}".format(org_shape))


            '''
            Here we are going to convert 'Price' ot Lacs metrics, which we use widely in Indian Market
            ‘df["Price"]=(df["Price"]/100000).round(1)‘

            As well as i am going to drop some columns like 'BED', 'VaastuCompliant', 'Microwave', 'GolfCourse', 'TV',
            'DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator','WashingMachine'
            '''

            dataframe["Price"]=(dataframe["Price"]/100000).round(1)
            logging.info("Price column has been converted")

            dataframe.drop(columns=['BED', 'VaastuCompliant', 'Microwave', 'GolfCourse', 'TV',
                            'DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator','WashingMachine'],inplace= True)
            logging.info("['BED', 'VaastuCompliant', 'Microwave', 'GolfCourse', 'TV','DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator','WashingMachine'] these columns has been dropped")

            s = dataframe.shape
            print("Shape of df after 10 columns removed: {}".format(s))

            dataframe.to_csv(self.transformation_config.raw_data01_path)

            return dataframe
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def Datasplit(self):
        try:
            df= pd.read_csv(self.transformation_config.raw_data01_path)

            logging.info("Train test split is initiated")
            train_set, test_set= train_test_split(df, test_size=0.2, random_state=42)

            print("Shape of train_set after 10 columns removed: {}".format(train_set.shape))
            print("Shape of test_set after 10 columns removed: {}".format(test_set.shape))

            train_set.to_csv(self.transformation_config.train_data_path, index= False, header= True)
            test_set.to_csv(self.transformation_config.test_data_path, index= False, header= True)

            return(self.transformation_config.train_data_path, self.transformation_config.test_data_path)
        except Exception as e:
            raise CustomException(e,sys)
        
    def Initiate_dataTransform(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)
        





        





