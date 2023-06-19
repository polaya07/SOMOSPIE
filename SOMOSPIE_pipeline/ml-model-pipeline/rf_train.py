def rf_train(data_path: str, maxtree: int, seed: int, out_model:str)-> str:
    # Libraries
    import numpy as np
    import json
    import pickle
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import mean_squared_error

    # Functions 
    def random_parameter_search(rf, x_train, y_train, maxtree, seed):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 300, stop = maxtree, num = 100)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]# Create the random grid
        params = {'n_estimators': n_estimators,
                    'max_features': ['sqrt'],
                    'max_depth': [20,50,70],
                    'bootstrap': [True],
                    'n_jobs':[-1]}
        # Random search based on the grid of params and n_iter controls number of random combinations it will try
        # n_jobs=-1 means using all processors
        # random_state sets the seed for manner of reproducibility 
        params_search = RandomizedSearchCV(rf, params, verbose=1, cv=10, n_iter=10, random_state=seed, n_jobs=-1)
        params_search.fit(x_train,y_train)
        # Check the results from the parameter search  
        print(params_search.best_score_)
        print(params_search.best_params_)
        print(params_search.best_estimator_)
        return params_search.best_estimator_

    def validate_rf(rf, x_test, y_test):
        # Predict on x_test
        y_test_predicted = rf.predict(x_test)
        # Measure the rmse
        rmse = np.sqrt(mean_squared_error(y_test, y_test_predicted))
        # Print error	
        #print("Predictions of soil moisture:", y_test_predicted)
        #print("Original values of soil moisture:", y_test)
        print("The rmse for the validation is:", rmse)

    # Start by reading the data
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

    # Define initial model
    rf = RandomForestRegressor()
    # Random parameter search for rf
    best_rf = random_parameter_search(rf, x_train, y_train, maxtree, seed)
    best_rf.fit(x_train, y_train)
    #Path(args.pathtomodel).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(best_rf, open(out_model, 'wb'))
    # Validate
    validate_rf(best_rf, x_test, y_test)
    return out_model




