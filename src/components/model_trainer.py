import os
import sys
from dataclasses import dataclass
import pandas as pd

#from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
#from xgboost import XGBRegressor

from sklearn.preprocessing import LabelEncoder

from src.exceptions import CustomException
from src.logger import logging

from src.utils import save_obj,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()
    def initiate_model_training(self,train_arr, test_arr):
        try:
            '''
            train_df= pd.read_csv(train_data_path)
            test_df= pd.read_csv(test_data_path)
            logging.info("data spliting")
            X_train, X_test, y_train, y_test=(
                train_df.drop(columns=["Price"],axis=1),
                test_df.drop(columns=["Price"],axis=1),
                train_df["Price"],
                test_df["Price"]
                )
            
            
            label_encoder = LabelEncoder()
            y_train= label_encoder.fit_transform(y_train)

            y_test= label_encoder.transform(y_test)


            
            print("dataframe has follwing shape {} & columns: {}".format(X_train.columns,X_train.shape))
            print("dataframe has follwing shape {} & columns: {}".format(y_train,y_train.shape))


            print("dataframe has follwing shape {} & columns: {}".format(X_test.columns,X_test.shape))
            print("dataframe has follwing shape {} & columns: {}".format(y_test,y_test.shape))
            '''
            
            print("train_arr has follwing shape {} & columns: {}".format(train_arr,train_arr.shape))
            print("test_Arr has follwing shape {} & columns: {}".format(test_arr,test_arr.shape))


            X_train, y_train, X_test, y_test=(
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            ) 
            logging.info("Read train and test data completed")
            
            print("################======================================###################")

            print("dataframe has follwing shape {} & columns: {}".format(X_train,X_train.shape))
            print("dataframe has follwing shape {} & columns: {}".format(y_train,y_train.shape))


            print("dataframe has follwing shape {} & columns: {}".format(X_test,X_test.shape))
            print("dataframe has follwing shape {} & columns: {}".format(y_test,y_test.shape))

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                #"XGBRegressor": XGBRegressor(),
                #"CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                #"PolynomialRegression":PolynomialFeatures(),
            }

            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test= X_test, y_test=y_test, models=models)
            
            print("Model scores: ")
            print(model_report)

            best_model_score= max(sorted(model_report.values()))

            best_model_name= list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model= models[best_model_name]
            print(best_model)

            if best_model_score< 0.6:
                print("No best model found")
                #raise CustomException("No best model found",)
            logging.info("Best found model on bith training and testing dataset")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted= best_model.predict(X_test)

            r2_squre= r2_score(y_test, predicted)
            return r2_squre

        except Exception as e:
            raise CustomException(e,sys)
            

        


