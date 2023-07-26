from typing import NamedTuple

def load_data(input_path: str,dir: str, out_data:str)-> NamedTuple('Output', [("data", str), 
                                                                            ('scaler', str)]):
    import numpy as np
    import json
    import pandas as pd
    import pickle
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from osgeo import gdal
    
    def get_band_names(raster):
        ds = gdal.Open(raster, 0)
        names = []
        for band in range(ds.RasterCount):
                b = ds.GetRasterBand(band + 1)
                names.append(b.GetDescription())
        ds = None
        return names

    def tif2df(raster_file):
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

        band_names = get_band_names(raster_file)

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
        # df.to_csv(output_file, index=None)
        return df

    print("Reading training data from", input_path)
    training_data = tif2df(input_path)
    print(training_data)
    
    x_train, x_test, y_train, y_test = train_test_split(training_data.loc[:,training_data.columns != 'z'], training_data.loc[:,'z'], test_size=0.1)
    #print(x_train,"\n",y_train,"TEST\n",x_test,y_test)
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    #print("SCALED")
    #print(x_train,"\n",y_train,"TEST\n",x_test,y_test)

    # Save scaler model so it can be reused for predicting
    pickle.dump(ss, open(dir+'scaler.pkl', 'wb'))

    # Save data to train with different ml-models
    data = {'x_train' : x_train.tolist(),
            'y_train' : y_train.tolist(),
            'x_test' : x_test.tolist(),
            'y_test' : y_test.tolist()}
    # Creates a json object based on `data`
    data_json = json.dumps(data)

    # Saves the json object into a file
    with open(out_data, 'w') as out_file:
        json.dump(data_json, out_file)

       # Create path to where data and scaler are saved
    
    return [out_data,dir+'scaler.pkl']