from flask import Flask, request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.components.pipeline.predict_pipeline import CustomData, PredictPipeline

application= Flask(__name__)

app=application

##Route for a Homepage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods= ['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        #this is actually the exacgt class which gets created in src/pipe_line/predict_pipeline.py
        data= CustomData(
            #Price= request.form.get('Price'),
            Area= request.form.get('Area'),
            No_of_Bedrooms= request.form.get('No_of_Bedrooms'),            
            MaintenanceStaff= request.form.get('MaintenanceStaff'),
            Gymnasium= request.form.get('Gymnasium'),
            SwimmingPool= request.form.get('SwimmingPool'),
            ClubHouse= request.form.get('ClubHouse'),
            Security= request.form.get('Security'),
            CarParking= request.form.get('CarParking'),
            Children_playarea= request.form.get('Children_playarea'),
            LiftAvailable= request.form.get('LiftAvailable'),
            City= request.form.get('City')
        )      # rerturns a DF &&& using request.form.get we are reading the data from home.html page


        pred_df= data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline= PredictPipeline()
        results= predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])  # resrults variable is used in home.html to show the predicted values


if __name__=="__main__":
    #app.run(host="0.0.0.0", debug=True)
    app.run(host="0.0.0.0", port=8080, debug=True)

