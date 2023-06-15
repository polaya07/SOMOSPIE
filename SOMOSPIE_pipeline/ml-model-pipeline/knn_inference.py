def knn_inference (eval_path: str):
    import numpy as np
    import pandas as pd
    import pickle

    print("Reading evaluation data from", eval_path)
    evaluation_data = pd.read_csv(eval_path+'eval.csv')
    # Load ss model
    ss = pickle.load(open(eval_path+'scaler.pkl', 'rb'))
    x_predict = ss.transform(evaluation_data)
    # Load knn regressor
    knn = pickle.load(open(eval_path+'model_knn.pkl', 'rb'))
    # Predict on evaluation data
    y_predict = knn.predict(x_predict)
    # Create dataframe with long, lat, soil moisture
    out_df = pd.DataFrame(data={'x':evaluation_data['x'].round(decimals=9), 'y':evaluation_data['y'].round(decimals=9), 'sm':y_predict})
    out_df = out_df.reindex(['x','y','sm'], axis=1)
    #Print to file predictions 
    out_df.to_csv(eval_path+"predictions_knn.csv", index=False, header=False)
