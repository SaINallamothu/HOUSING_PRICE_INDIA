import pandas as pd
import numpy as ny
import os
import sys

import dill
import pickle

from src.logger import logging
from src.exceptions import CustomException
from sklearn.metrics import r2_score

def load_csv(folderpath: str):
    try:
        dateframe= []
        for filename in os.listdir(folderpath):
            if filename.endswith(".csv"):
                file_path= os.path.join(folderpath,filename)
                
                df = pd.read_csv(file_path)

                df["City"]=filename[:-4]  #adding a new column City with filename

                dateframe.append(df)

        combined_df= pd.concat(dateframe, ignore_index= True)
        return combined_df
    except Exception as e:
        raise CustomException(e, sys)
    
# TO SAVE THE PKL OBJ
def save_obj(file_path, obj):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)

# TO LOAD THE MODEL.PKL PICKLE FILE TO REUSE     
def load_obj(file_path):
    try:
        with open(file_path, "rb") as obj:
            return pickle.load(obj)
    except Exception as e:
        raise CustomException(e,sys)

#TO EVALUATE THE MODEL
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report={}
        for i in range(len(list(models))):
            model= list(models.values())[i]

            model.fit(X_train,y_train)

            y_train_pred= model.predict(X_train)
            y_test_pred= model.predict(X_test)

            train_model_score= r2_score(y_train, y_train_pred)
            test_model_score= r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]]= test_model_score 
        return report
    except Exception as e:
        raise CustomException(e,sys)
    


