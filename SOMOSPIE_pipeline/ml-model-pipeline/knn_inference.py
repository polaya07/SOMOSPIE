
def knn_inference (model_path: str, scaler_path:str, eval_path:str,  out_dir:str, predictions:str, band_names:list)-> str:
    import numpy as np
    import pandas as pd
    import pickle  
    from osgeo import gdal, ogr  
    import sklearn


    def tif2df(raster_file, band_names) :
        ds = gdal.Open(raster_file, 0)
        xmin, res, _, ymax, _, _ = ds.GetGeoTransform()
        xsize = ds.RasterXSize
        ysize = ds.RasterYSize
        xstart = xmin + res / 2
        ystart = ymax - res / 2

        x = np.arange(xstart, xstart + xsize * res, res)
        y = np.arange(ystart, ystart - ysize * res, -res)
        x = np.tile(x[:xsize], ysize)
        y = np.repeat(y[:ysize], xsize)

        n_bands = ds.RasterCount
        bands = np.zeros((x.shape[0], n_bands))
        for k in range(1, n_bands + 1):
            band = ds.GetRasterBand(k)
            data = band.ReadAsArray()
            data = np.ma.array(data, mask=np.equal(data, band.GetNoDataValue()))
            data = data.filled(np.nan)
            bands[:, k-1] = data.flatten()

        column_names = ['x', 'y'] + band_names
        stack = np.column_stack((x, y, bands))
        df = pd.DataFrame(stack, columns=column_names)
        df.dropna(inplace=True)
        #print(df.describe(include='all'))
        #print("5 ENTRIES\n",df.head())
        #print("Size of the df\n",df.size)
        print(df.info())
        #df.to_csv(output_file, index=None)
        return df

    print("Reading evaluation data from", eval_path)
    evaluation_data = tif2df(eval_path, band_names)
    #evaluation_data = eval_path
    # Load ss model
    ss = pickle.load(open(scaler_path, 'rb'))
    x_predict = ss.transform(evaluation_data)
    # Load knn regressor
    knn = pickle.load(open(model_path, 'rb'))
    # Predict on evaluation data
    y_predict = knn.predict(x_predict)
    # Create dataframe with long, lat, soil moisture
    out_df = pd.DataFrame(data={'x':evaluation_data['x'].round(decimals=9), 'y':evaluation_data['y'].round(decimals=9), 'sm':y_predict})
    out_df = out_df.reindex(['x','y','sm'], axis=1)
    #Print to file predictions 
    import os
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_df.to_csv(predictions, index=False, header=False)
    return predictions
