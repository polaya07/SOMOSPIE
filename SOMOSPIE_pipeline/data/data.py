## Script for getting the training data

import numpy as np
import json
import argparse
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

def get_parser():
    parser = argparse.ArgumentParser(description='Arguments and data files for executing Nearest Neighbors Regression.')
    parser.add_argument('-t', "--trainingdata", help='Training data')
    #parser.add_argument('-m', "--pathtomodel", help='Directory where the knn model will be saved')
    return parser

def load_data(args):
    print("Reading training data from", args.trainingdata)
        # Gets and split dataset
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Creates `data` structure to save and 
    # share train and test datasets.
    data = {'x_train' : x_train.tolist(),
            'y_train' : y_train.tolist(),
            'x_test' : x_test.tolist(),
            'y_test' : y_test.tolist()}

    # Creates a json object based on `data`
    data_json = json.dumps(data)

    # Saves the json object into a file
    with open(args.trainingdata, 'w') as out_file:
        json.dump(data_json, out_file)

if __name__ == "__main__":
    parser=get_parser()
    args = parser.parse_args()
    load_data (args)
