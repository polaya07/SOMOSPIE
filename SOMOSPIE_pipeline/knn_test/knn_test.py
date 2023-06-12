#!/usr/bin/env python3

## Script KKNN 
## PREPARED BY PAULA OLAYA (UTK) TO MICHELA TAUFER, 2021
## Modified by Leobardo Valera, 2021

## Commandline example: 
## ./2b-kknn.py -t ../data/2012/t-postproc/6.2.csv -e ../data/2012/e-postproc/6.2.csv -o ../data/2012/example.csv -k 20

import numpy as np
import pandas as pd
import argparse
import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error


#Input arguments to execute the k-Nearest Neighbors Regression 
def get_parser():
    parser = argparse.ArgumentParser(description='Arguments and data files for executing Nearest Neighbors Regression.')
    parser.add_argument('-t', "--test", help='Passing data')
    parser.add_argument('-e', "--evaluationdata", help='Evaluation data')
    parser.add_argument('-o', "--outputdata", help='Predictions')
    return parser 

#Translate from namespaces to Python variables 
def knn_inference (args):	
    print("Reading evaluation data from", args.evaluationdata)
    evaluation_data = pd.read_csv(args.evaluationdata+'eval.csv')
    # Load ss model
    ss = pickle.load(open(args.evaluationdata+'scaler.pkl', 'rb'))
    x_predict = ss.transform(evaluation_data)
    # Load knn regressor
    knn = pickle.load(open(args.evaluationdata+'model.pkl', 'rb'))
    # Predict on evaluation data
    y_predict = knn.predict(x_predict)
    # Create dataframe with long, lat, soil moisture
    out_df = pd.DataFrame(data={'x':evaluation_data['x'].round(decimals=9), 'y':evaluation_data['y'].round(decimals=9), 'sm':y_predict})
    out_df = out_df.reindex(['x','y','sm'], axis=1)
    #Print to file predictions 
    out_df.to_csv(args.evaluationdata+"predictions.csv", index=False, header=False)


if __name__ == "__main__":	
    parser=get_parser()
    args = parser.parse_args()
    knn_inference (args)
