import sys
import pandas as pd
import os

from src.exceptions import CustomException
from src.utils import load_obj

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, feature):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model= load_obj(file_path= model_path)
            preprocessor= load_obj(file_path= preprocessor_path)
            data_scaled= preprocessor.transform(feature)
            preds= model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

## using this class we are going to mapp whatever the input values captured by home.html to our self
class CustomData():
    def __init__(self,
        #Price: int,
        Area: int,
        No_of_Bedrooms: int,
        MaintenanceStaff: int,
        Gymnasium: int,
        SwimmingPool: int,
        ClubHouse: int,
        Security: int,
        CarParking: int,
        Children_playarea: int,
        LiftAvailable: int,
        City: str):
        #self.Price= Price
        self.Area=Area
        self.No_of_Bedrooms=No_of_Bedrooms
        self.MaintenanceStaff= MaintenanceStaff
        self.Gymnasium=Gymnasium
        self.SwimmingPool=SwimmingPool
        self.ClubHouse=ClubHouse
        self.Security=Security
        self.CarParking=CarParking
        self.Children_playarea=Children_playarea
        self.LiftAvailable=LiftAvailable
        self.City=City

    ## Here we are going to convert the variables to a DF, sicne our model was trainned using DF
    def get_data_as_data_frame(self):

        try:
            custom_data_input_dict={
                        #"Price": [self.Price],
                        "Area": [self.Area],
                        "No_of_Bedrooms": [self.No_of_Bedrooms],
                        "MaintenanceStaff": [self.MaintenanceStaff],
                        "Gymnasium": [self.Gymnasium],
                        "SwimmingPool": [self.SwimmingPool],
                        "ClubHouse": [self.ClubHouse],
                        "Security": [self.Security],
                        "CarParking": [self.CarParking],
                        "Children_playarea": [self.Children_playarea],
                        "LiftAvailable": [self.LiftAvailable],
                        "City": [self.City]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
        
