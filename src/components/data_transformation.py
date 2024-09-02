import sys
import os
import pandas as pd
import numpy as np

from src.logger import logging
from src.exceptions import CustomException
from src.utils import save_obj

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer #used to create a pipeline to do tranformation
from sklearn.impute import SimpleImputer #to deal with missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler, OrdinalEncoder
from category_encoders import cat_boost,CatBoostEncoder

from dataclasses import dataclass

@dataclass
class DatatransformationConfig:
    raw_data01_path: str=os.path.join("artifacts/CSV_files", "raw_data01.csv")
    train_data_path: str= os.path.join("artifacts/CSV_files", "train_data.csv")
    test_data_path: str= os.path.join("artifacts/CSV_files", "test_data.csv")
    preprocessor_obj_file_path: str= os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config=DatatransformationConfig()

    def DataDropping(self, dataframe):
        try:
            org_shape = dataframe.shape
            print("Shape of df before columns removed: {}".format(org_shape))


            '''
            Here we are going to convert 'Price' ot Lacs metrics, which we use widely in Indian Market
            ‘df["Price"]=(df["Price"]/100000).round(1)‘

            As well as i am going to drop some columns like 'BED', 'VaastuCompliant', 'Microwave', 'GolfCourse', 'TV',
            'DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator','WashingMachine'
            '''
            dataframe["Price"]=(dataframe["Price"]/100000).round(1)
            logging.info("Price column has been converted")
            """
            These are the columns getting removed from raw dataframe: 
            [ 'Resale', 'LandscapedGardens','JoggingTrack', 'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall',
                'Intercom', 'SportsFacility', 'ATM', 'School','PowerBackup', 'StaffQuarter',
                'Cafeteria', 'MultipurposeRoom', 'Hospital', 'WashingMachine',
                'Gasconnection', 'AC', 'Wifi','BED', 'VaastuCompliant', 'Microwave', 'GolfCourse', 'TV',
                'DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator', 'City']
            """




            dataframe.drop(columns=['Resale', 'LandscapedGardens','JoggingTrack', 'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall',
                'Intercom', 'SportsFacility', 'ATM', 'School','PowerBackup', 'StaffQuarter',
                'Cafeteria', 'MultipurposeRoom', 'Hospital', 'WashingMachine',
                'Gasconnection', 'AC', 'Wifi','BED', 'VaastuCompliant', 'Microwave', 'GolfCourse', 'TV',
                'DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator'],inplace= True)
            logging.info("'Resale', 'LandscapedGardens','JoggingTrack', 'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall','Intercom', 'SportsFacility', 'ATM', 'School','PowerBackup', 'StaffQuarter','Cafeteria', 'MultipurposeRoom', 'Hospital', 'WashingMachine','Gasconnection', 'AC', 'Wifi','BED', 'VaastuCompliant', 'Microwave', 'GolfCourse', 'TV', 'DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator', 'City' these columns has been dropped")

            s = dataframe.shape
            print("Shape of df after columns removed: {}".format(s))

            dataframe.to_csv(self.transformation_config.raw_data01_path,index=False)

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

    def get_data_transformer_object(self):
        try:
            numerical_columns= [ 'Area', 'No. of Bedrooms', 'MaintenanceStaff', 'Gymnasium','SwimmingPool', 'ClubHouse', '24X7Security', 'CarParking',"Children'splayarea", 'LiftAvailable']
            categorical_columns=[ 'City','Location']

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                #("one_hot_encoder",OneHotEncoder(handle_unknown='ignore')), #if the test data has new categories like new location names in test data which are not in train, then it will ignore that, instead of raising a Value error 
                ("cat-boost-encoder",cat_boost.CatBoostEncoder(handle_missing="value", handle_unknown="value")),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns standatrd scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)


    def Initiate_data_transformation(self):
        try:


            train_df= pd.read_csv(self.transformation_config.train_data_path)
            test_df= pd.read_csv(self.transformation_config.test_data_path)
            print("dataframe has follwing shape {} & columns: {}".format(train_df.columns,train_df.shape))
            print("dataframe has follwing shape {} & columns: {}".format(test_df.columns,test_df.shape))
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj= self.get_data_transformer_object()

            target_column_name= "Price"
            numerical_columns= ['Area', 'No. of Bedrooms', 'MaintenanceStaff', 'Gymnasium','SwimmingPool', 'ClubHouse', '24X7Security', 'CarParking',"Children'splayarea", 'LiftAvailable']

            input_feature_train_df= train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df= train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df= test_df[target_column_name]

            print("test")
            print(input_feature_test_df.shape)
            print(target_feature_test_df.shape)

            print("train")
            print(input_feature_train_df.shape)
            print(target_feature_train_df.shape)

            print(input_feature_train_df.head(5), target_feature_test_df.head(5))

            logging.info("Applying preprocessing tranformation on train and test dataframes")

           
            target_feature_train_arr= np.array(target_feature_train_df)
            target_feature_test_arr= np.array(target_feature_test_df)

            input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df,target_feature_train_arr)
            input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)
            
            print("train arr")
            print(input_feature_train_arr.shape)
            print(target_feature_train_arr.shape)

            target_feature_train_arr=target_feature_train_arr.reshape(-1, 1) 
            print("reshaped train arr")
            print(target_feature_train_arr.shape)


            train_arr = np.concatenate((input_feature_train_arr, target_feature_train_arr.reshape(-1, 1)), axis=1)
            print(train_arr)
            #train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            logging.info(f"Saved preprocessing object")

            #careating a pickle file using save_object from src.utils
            print(self.transformation_config.preprocessor_obj_file_path)
            save_obj(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,test_arr, self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        





        





