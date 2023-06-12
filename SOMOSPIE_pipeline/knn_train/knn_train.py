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
    parser.add_argument('-d', "--data", help='Data')
    parser.add_argument('-m', "--pathtomodel", help='Directory where the knn model will be saved')
    parser.add_argument('-k', "--maxK", help='Mamximum k to try for finding optimal model', default=20)
    parser.add_argument('-seed', "--seed", help='Seed for reproducibility purposed in random research grid', default=3)
    return parser 

#Translate from namespaces to Python variables 
def knn_train (args):	
    print("Reading training data from", args.trainingdata)
        # Open and reads file "data"
    with open(args.data+"data.json") as data_file:
        data = json.load(data_file)

    data = json.loads(data)

    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    #print(training_data)
    maxK = int(args.maxK)
    seed = int(args.seed)

        # Define initial model
    knn = KNeighborsRegressor()
    # Random parameter search of n_neighbors, weigths and metric
    best_params = random_parameter_search(knn, x_train, y_train, maxK, seed)
    # Based on selection build the new regressor
    knn = KNeighborsRegressor(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'],
    				metric=best_params['metric'], n_jobs=-1)
    # Fit the new model to data
    knn.fit(x_train, y_train)
    # Save model
    Path(args.pathtomodel).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(knn, open(args.data+'model.pkl', 'wb'))

    # Validate
    validate_knn(knn, x_test, y_test)


def random_parameter_search(knn, x_train, y_train, maxK, seed):
    # Dictionary with all the hyperparameter options for the knn model: n_neighbors, weights, metric
    params = {'n_neighbors': list(range(2,maxK)),
    	  'weights': ['uniform','distance', gaussian],
    	  'metric': ['euclidean','minkowski']
             }
    # Random search based on the grid of params and n_iter controls number of random combinations it will try
    # n_jobs=-1 means using all processors
    # random_state sets the seed for manner of reproducibility 
    params_search = RandomizedSearchCV(knn, params, verbose=1, cv=10, n_iter=50, random_state=seed, n_jobs=-1)
    params_search.fit(x_train,y_train)
    # Check the results from the parameter search  
    print(params_search.best_score_)
    print(params_search.best_params_)
    print(params_search.best_estimator_)
    return params_search.best_params_

def gaussian(dist, sigma = 4):
    # Input a distance and return its weight using the gaussian kernel 
    weight = np.exp(-dist**2/(2*sigma**2))
    return weight


def validate_knn(knn, x_test, y_test):
    # Predict on x_test
    y_test_predicted = knn.predict(x_test)
    # Measure the rmse
    rmse = np.sqrt(mean_squared_error(y_test, y_test_predicted))
    # Print error	
    #print("Predictions of soil moisture:", y_test_predicted)
    #print("Original values of soil moisture:", y_test)
    print("The rmse for the validation is:", rmse)


if __name__ == "__main__":	
    parser=get_parser()
    args = parser.parse_args()
    knn = train_knn(args)

