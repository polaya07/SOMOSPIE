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