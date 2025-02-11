def knn_train (data_path: str, k: int, seed: int, out_model:str)-> str:
    import numpy as np
    import pickle
    import json
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import mean_squared_error
    import os

    # Define functions for training
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
    
    print("Checking if model ", out_model, " exists")
    if os.path.isfile(out_model):
        print("The model ", out_model, " exists")
        return out_model
    else:
        print("Reading training data from", data_path)
            # Open and reads file "data"
        with open(data_path) as data_file:
            data = json.load(data_file)

        data = json.loads(data)

        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']
        #print(training_data)
        maxK = int(k)
        seed = int(seed)

            # Define initial model
        knn = KNeighborsRegressor(leaf_size=3000)
        # Random parameter search of n_neighbors, weigths and metric
        best_params = random_parameter_search(knn, x_train, y_train, maxK, seed)
        # Based on selection build the new regressor
        knn = KNeighborsRegressor(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'],
                        metric=best_params['metric'], n_jobs=-1)
        # Fit the new model to data
        knn.fit(x_train, y_train)
        # Save model
        pickle.dump(knn, open(out_model, 'wb'))

        # Validate
        validate_knn(knn, x_test, y_test)
        return out_model