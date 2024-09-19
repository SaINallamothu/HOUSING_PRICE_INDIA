# HOUSING_PRICE_INDIA

## Objective:

Objective of this project is to predict a Price of Home in a given City, basing the values of independent variables like City, Area, No. of Bedrooms, Lift Availability, Car parking, Security, Children play area, Clubhouse, Maintenance Staff, Swimming pool.
## Tech Stack

Here are the technologies I use:

<p align="left">
    <img src="https://github.com/devicons/devicon/blob/v2.16.0/icons/python/python-original.svg" alt="Python" width="40" height="40"/>
  <img src="https://github.com/devicons/devicon/blob/v2.16.0/icons/html5/html5-original.svg" alt="html" width="40" height="40"/>
    <img src="https://github.com/devicons/devicon/blob/v2.16.0/icons/flask/flask-original-wordmark.svg" alt="Flask" width="40" height="40"/>  
  <img src="https://github.com/devicons/devicon/blob/v2.16.0/icons/docker/docker-original.svg" alt="Docker" width="40" height="40"/>
  <img src="https://github.com/devicons/devicon/blob/v2.16.0/icons/githubactions/githubactions-original.svg" alt="GitHub Actions" width="40" height="40"/>
    <img src="https://github.com/devicons/devicon/blob/v2.16.0/icons/amazonwebservices/amazonwebservices-original-wordmark.svg" alt="AWS" width="40" height="40">
</p>


### Project Structure

```bash
.
├── Dockerfile
├── Notebook
│   └── EDA.ipynb
├── README.md
├── application.py
├── artifacts
│   ├── CSV_files
│   │   ├── raw_data.csv
│   │   ├── raw_data01.csv
│   │   ├── test_data.csv
│   │   └── train_data.csv
│   ├── base_files
│   │   ├── Bangalore.csv
│   │   ├── Chennai.csv
│   │   ├── Delhi.csv
│   │   ├── Hyderabad.csv
│   │   ├── Kolkata.csv
│   │   └── Mumbai.csv
│   ├── model.pkl
│   └── preprocessor.pkl
├── logs
├── requirements.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── __pycache__
│   ├── components
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── pipeline
│   │       ├── __init__.py
│   │       ├── __pycache__
│   │       └── predict_pipeline.py
│   ├── exceptions.py
│   ├── logger.py
│   └── utils.py
└── templates
    ├── home.html
    └── index.html
```

## Implementation:

1.	Data source:
Datasets for training of this model were taken from Kaggle, here is the link:https://www.kaggle.com/datasets/ruchi798/housing-prices-in-metropolitan-areas-of-india

2.	Data Extraction & Transformation:
i.	Source of the data shows all individual csv files without City column in them, we are going to add a new column ‘City’ from its file name, before we are going to combining them into single csv (‘artifacts/CSV_files/raw_data.csv’). This done using the custom function written in ‘utils.py’. 
ii.	Since the raw data consists of lot of unwanted columns, we are going to do transformation for few and remove the remaining unwanted columns from the raw_data.csv and save the new version of it into ‘raw_data01.csv’.

Here we are going to convert 'Price' ot Lacs metrics, which we use widely in Indian Market
            ‘df["Price"]=(df["Price"]/100000).round(1)‘

These are the columns getting removed from raw dataframe: 
            [ 'Resale', 'LandscapedGardens','JoggingTrack', 'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall', 'Intercom', 'SportsFacility', 'ATM', 'School','PowerBackup', 'StaffQuarter', 'Cafeteria', 'MultipurposeRoom', 'Hospital', 'WashingMachine', 'Gasconnection', 'AC', 'Wifi','BED', 'VaastuCompliant', 'Microwave', 'GolfCourse', 'TV', 'DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator', 'City']

iii.	Next step would be splitting the data into train & test. Test size of the data would be 20% of actual data points from raw_data01.csv. After the splitting is done, saved those data points into csv files (‘artifacts/CSV_files/’). 
iv.	Now, it is the time to do some feature engineering.
We used simple imputer with a strategy of ‘median’ for numerical columns and ‘most frequent’ for categorical columns. Coming to encoding, using target encoding method for categorical columns (‘City’), one hot encoding is also useful as per my trails.
v.	Once we fit this transformation with training data, have saved this model in pickle file for further use.
3.	Model Training:

Our idea of model training is bit different, here want to train different prediction models and evaluate the metrics of efficiency of the model like r2_square. 

Here are the list of models, that we have taken into account:
"Random Forest, Decision Tree, Gradient Boosting, Linear Regression, 
KneighborsRegressor, AdaBoost Regressor“. 

Once train the model, we evaluate the model efficiency with r2_Squre. Whatever the model gives the higher r2_square value, we choose that model to be the one used further for Housing Price prediction.

Model has been saved into the pickle format, and it will be used in flask application.

4.	Model deployment:
We are going to deploy this model using Flask framework, in an EC2 (with Ubuntu image) instance on AWS. In order to deploy this model, we have to make sure that the application.py is not set to be in debug mood (debug= False).



## CI CD Pipeline

Continuous Integration is achieved using GitHub Actions, we have to write a new workflow configuration under ‘.github/workflows/main.yml’. This yml file defines the sequential process of implementing our CI CD pipeline from Integration, testing, pushing Docker image to AWS ECR, to deployment of application. 

There are some variables to authenticate the AWS login and ECR repository details. We must save them under Git Repository settings/ security/actions as variables. So, it will help to protect our Private key & access key from being stolen, when we share the code with someone. As mentioned before these are the variables, which we use in our workflow main.yml file. 

Once we finish with these steps, now have to go to AWS and lauch EC2 instance with any standard OS (I would recommend Ubuntu). To finish the automation, finally have to launch a self-hosted runner from GitHub repo/settings/actions/runners, then there will be set off commands to run in our EC2 CLI to establish a connection between EC2 and Git Repo. 

If you are using Ubuntu in your EC2, here are the set off commands (this may vary in future or if you select another OS on EC2):

Download
### Create a folder
$ mkdir actions-runner && cd actions-runner
### Download the latest runner package
$ curl -o actions-runner-linux-x64-2.319.1.tar.gz -L https://github.com/actions/runner/releases/download/v2.319.1/actions-runner-linux-x64-2.319.1.tar.gz
### Optional: Validate the hash
$ echo "3f6efb7488a183e291fc2c62876e14c9ee732864173734facc85a1bfb1744464 actions-runner-linux-x64-2.319.1.tar.gz" | shasum -a 256 -c
### Extract the installer
$ tar xzf ./actions-runner-linux-x64-2.319.1.tar.gz
Configure
### Create the runner and start the configuration experience
$ ./config.sh --url https://github.com/SaINallamothu/HOUSING_PRICE_INDIA --token APKML6636VIKMJ7FWVRNAOLG5M5FI
### Last step, run it!
$ ./run.sh


Accessing our Application on EC2:

Make sure you have IPv4 enabled while launching an EC2 instance. Since we are using port: 8080, now copy the IPv4 and paste it in your Browser along with 8080 port.

Example: 'IPv4:8080'

There is another networking specific configuration to do it on AWS under security-group Inbound rules, are allowing all traffic with following protocol and port.

SSH		22
Custom TCP	8080

Thank you for your attention
