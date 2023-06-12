## Script for getting the training data

import numpy as np
import json
import argparse
import pandas as pd
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def get_parser():
    parser = argparse.ArgumentParser(description='Arguments and data files for executing Nearest Neighbors Regression.')
    parser.add_argument('-t', "--trainingdata", help='Training data')
    parser.add_argument('-m', "--pathtomodel", help='Directory where the knn model will be saved')
    return parser

def load_data(args):
    print("Reading training data from", args.trainingdata)
    print("Current directory:", Path.cwd(), " and full path ", Path(__file__).resolve().parent)
    print("Files and directory in the bucket", os.listdir(Path.cwd()))
    training_data = pd.read_csv(args.trainingdata+"train.csv")
    col = list(training_data.columns)
    col[2] = 'z'
    training_data.columns = col
    # Create path to where data and scaler are saved
    print("Creating directory:", args.pathtomodel, os.path.abspath(args.pathtomodel))
    Path(args.pathtomodel).parent.mkdir(parents=True, exist_ok=True)
    print("Output location", os.listdir(args.pathtomodel))
    
    x_train, x_test, y_train, y_test = train_test_split(training_data.loc[:,training_data.columns != 'z'], training_data.loc[:,'z'], test_size=0.1)
    #print(x_train,"\n",y_train,"TEST\n",x_test,y_test)
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    #print("SCALED")
    #print(x_train,"\n",y_train,"TEST\n",x_test,y_test)

    # Save scaler model so it can be reused for predicting
    pickle.dump(ss, open(args.trainingdata+'scaler.pkl', 'wb'))

    # Save data to train with different ml-models
    data = {'x_train' : x_train.tolist(),
            'y_train' : y_train.tolist(),
            'x_test' : x_test.tolist(),
            'y_test' : y_test.tolist()}
    # Creates a json object based on `data`
    data_json = json.dumps(data)

    # Saves the json object into a file
    with open(args.trainingdata+"data.json", 'w') as out_file:
        json.dump(data_json, out_file)

if __name__ == "__main__":
    parser=get_parser()
    args = parser.parse_args()
    load_data (args)
