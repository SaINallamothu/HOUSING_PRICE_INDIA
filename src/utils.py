import pandas as pd
import numpy as ny
import os
import sys

from src.logger import logging
from src.exceptions import CustomException

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
    

