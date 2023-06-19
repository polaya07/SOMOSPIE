
def rf_inference(model_path: str, scaler_path: str, eval_path: str, predictions:str) -> str:
    # Libraries
    import numpy as np
    import pandas as pd
    import pickle

    print("Reading evaluation data from", eval_path)
    evaluation_data = pd.read_csv(eval_path)
    # Load ss model
    ss = pickle.load(open(scaler_path, 'rb'))
    x_predict = ss.transform(evaluation_data)
    # Load knn regressor
    rf = pickle.load(open(model_path, 'rb'))
    # Predict on evaluation data
    y_predict = rf.predict(x_predict)
    # Create dataframe with long, lat, soil moisture
    out_df = pd.DataFrame(data={'x':evaluation_data['x'].round(decimals=9), 'y':evaluation_data['y'].round(decimals=9), 'sm':y_predict})
    out_df = out_df.reindex(['x','y','sm'], axis=1)
    # Print to file predictions 
    out_df.to_csv(predictions, index=False, header=False)
    return predictions