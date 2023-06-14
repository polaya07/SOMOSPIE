import kfp
from kfp import dsl
from kfp.components import func_to_container_op


def load_data(input_path: str)-> str:
    import numpy as np
    import json
    import argparse
    import pandas as pd
    from pathlib import Path
    import pickle
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import os
    
    print("Reading training data from", input_path)
    print("Current directory:", Path.cwd(), " and full path ", Path(__file__).resolve().parent)
    print("Files and directory in the bucket", os.listdir(Path.cwd()))
    training_data = pd.read_csv(input_path+"train.csv")
    col = list(training_data.columns)
    col[2] = 'z'
    training_data.columns = col
 
    
    x_train, x_test, y_train, y_test = train_test_split(training_data.loc[:,training_data.columns != 'z'], training_data.loc[:,'z'], test_size=0.1)
    #print(x_train,"\n",y_train,"TEST\n",x_test,y_test)
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    #print("SCALED")
    #print(x_train,"\n",y_train,"TEST\n",x_test,y_test)

    # Save scaler model so it can be reused for predicting
    pickle.dump(ss, open(input_path+'scaler.pkl', 'wb'))

    # Save data to train with different ml-models
    data = {'x_train' : x_train.tolist(),
            'y_train' : y_train.tolist(),
            'x_test' : x_test.tolist(),
            'y_test' : y_test.tolist()}
    # Creates a json object based on `data`
    data_json = json.dumps(data)

    # Saves the json object into a file
    with open(input_path+"data.json", 'w') as out_file:
        json.dump(data_json, out_file)

       # Create path to where data and scaler are saved
    
    return input_path

#Translate from namespaces to Python variables 
def knn_train (data_path: str, k: int, seed: int)-> str:
    import numpy as np
    import pandas as pd
    import argparse
    import pickle
    import json
    from pathlib import Path
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import mean_squared_error

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
    
    print("Reading training data from", data_path)
        # Open and reads file "data"
    with open(data_path+"data.json") as data_file:
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
    knn = KNeighborsRegressor()
    # Random parameter search of n_neighbors, weigths and metric
    best_params = random_parameter_search(knn, x_train, y_train, maxK, seed)
    # Based on selection build the new regressor
    knn = KNeighborsRegressor(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'],
    				metric=best_params['metric'], n_jobs=-1)
    # Fit the new model to data
    knn.fit(x_train, y_train)
    # Save model

    #Path(args.pathtomodel).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(knn, open(data_path+'model.pkl', 'wb'))

    # Validate
    validate_knn(knn, x_test, y_test)
    return data_path

#Translate from namespaces to Python variables 
def knn_inference (eval_path: str):
    import numpy as np
    import pandas as pd
    import argparse
    import pickle
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import mean_squared_error
    print("Reading evaluation data from", eval_path)
    evaluation_data = pd.read_csv(eval_path+'eval.csv')
    # Load ss model
    ss = pickle.load(open(eval_path+'scaler.pkl', 'rb'))
    x_predict = ss.transform(evaluation_data)
    # Load knn regressor
    knn = pickle.load(open(eval_path+'model.pkl', 'rb'))
    # Predict on evaluation data
    y_predict = knn.predict(x_predict)
    # Create dataframe with long, lat, soil moisture
    out_df = pd.DataFrame(data={'x':evaluation_data['x'].round(decimals=9), 'y':evaluation_data['y'].round(decimals=9), 'sm':y_predict})
    out_df = out_df.reindex(['x','y','sm'], axis=1)
    #Print to file predictions 
    out_df.to_csv(eval_path+"predictions.csv", index=False, header=False)

@dsl.pipeline(name='somospiepipeline', description='Pipeline for somospie')
def pipeline():

    # Create a PVC where input data is stored
    pvc_op = dsl.VolumeOp(name="odh-pvc",
                           resource_name="pvc-odh-oklahoma1km",
                           storage_class="ibmc-s3fs-standard-regional",
                           size="10Gi",
                           annotations={"ibm.io/auto-create-bucket": "false",
                           "ibm.io/auto-delete-bucket": "false",
                           "ibm.io/bucket": "odh-oklahoma1km", 
                           "ibm.io/endpoint": "https://s3.us-east.cloud-object-storage.appdomain.cloud",
                           "ibm.io/secret-name": "po-secret"},
                           modes=dsl.VOLUME_MODE_RWO)

    data_op = kfp.components.create_component_from_func(load_data, base_image = "olayap/somospie")#packages_to_install=["numpy", "pandas", "scikit-learn"])
    knn_train_op = kfp.components.create_component_from_func(knn_train,  base_image = "olayap/somospie")#packages_to_install=["numpy", "pandas", "scikit-learn"])
    knn_inference_op = kfp.components.create_component_from_func(knn_inference,  base_image = "olayap/somospie")#packages_to_install=["numpy", "pandas", "scikit-learn"])
    
     # Get data and split in train and validation
    data_task = data_op("/cos/").add_pvolumes({"/cos/": pvc_op.volume})
    print(data_task.output)

    # Train and Test after splitting the data
    knn_train_task = knn_train_op(data_task.output, 20, 3).add_pvolumes({"/cos/": pvc_op.volume})
    print(knn_train_task.output)
    knn_inference_task = knn_inference_op(knn_train_task.output).add_pvolumes({"/cos/": pvc_op.volume})

if __name__ == '__main__':
    from kfp_tekton.compiler import TektonCompiler
    TektonCompiler().compile(pipeline, 'pipeline2.yaml')
    # kfp.Client().create_run_from_pipeline_func(basic_pipeline, arguments={})
 