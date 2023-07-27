
def inference(model_path: str, scaler_path: str, eval_path: str, out_dir:str, predictions:str, tmp_pred:str) -> str:
    import numpy as np
    import pickle  
    from osgeo import gdal, ogr  
    import sklearn
    import os
    import pathlib

    def get_band_names(raster):
        ds = gdal.Open(raster, 0)
        names = []
        for band in range(ds.RasterCount):
            b = ds.GetRasterBand(band + 1)
            names.append(b.GetDescription())
        ds = None
        return names

    def load_ds(evaluation_file, scaler_file):
        def tif2arr(raster_file):
            ds = gdal.Open(raster_file, 0)
            xmin, res, _, ymax, _, _ = ds.GetGeoTransform()
            xsize = ds.RasterXSize
            ysize = ds.RasterYSize
            xstart = xmin + res / 2
            ystart = ymax - res / 2

            x = np.arange(xstart, xstart + xsize * res, res, dtype=np.single)
            y = np.arange(ystart, ystart - ysize * res, -res,dtype=np.single)
            x = np.tile(x[:xsize], ysize)
            y = np.repeat(y[:ysize], xsize)

            n_bands = ds.RasterCount
            data = np.zeros((x.shape[0], n_bands), dtype=np.single)
            for k in range(1, n_bands + 1):
                band = ds.GetRasterBand(k)
                data[:, k-1] = band.ReadAsArray().flatten().astype(np.single)
                
            data = np.column_stack((x, y, data))
            del x, y
            data = data[~np.isnan(data).any(axis=1)]
            return data
    
        evaluation_data = tif2arr(evaluation_file) 
        ss = pickle.load(open(scaler_file, 'rb'))
        x_predict = ss.transform(evaluation_data)
        evaluation_data = evaluation_data[:,0:2]
        return x_predict, evaluation_data

    def rasterize(input_file, output_file, xres, yres):
        # When there is not a regular grid (has missing values)
        vrt_file = output_file[:-4] + '.vrt'
        if os.path.exists(vrt_file):
            os.remove(vrt_file)
            
        f = open(vrt_file, 'w')
        f.write('<OGRVRTDataSource>\n \
        <OGRVRTLayer name="{}">\n \
            <SrcDataSource>{}</SrcDataSource>\n \
            <GeometryType>wkbPoint</GeometryType>\n \
            <GeometryField encoding="PointFromColumns" x="x" y="y" z="z"/>\n \
        </OGRVRTLayer>\n \
    </OGRVRTDataSource>'.format('predictions', input_file)) # https://gdal.org/programs/gdal_grid.html#gdal-grid
        f.close()
        
        rasterize_options = gdal.RasterizeOptions(xRes=xres, yRes=yres, attribute='z', noData=np.nan, outputType=gdal.GDT_Float32, creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'], callback=gdal.TermProgress_nocb)
        r = gdal.Rasterize(output_file, vrt_file, options=rasterize_options)
        r = None
        os.remove(vrt_file)
    
    def predict(x_predict, evaluation_data, out_file, model_file):
        model = pickle.load(open(model_file, 'rb'))
        # Predict on evaluation data
        y_predict = model.predict(x_predict)
        
        evaluation_data = np.column_stack((evaluation_data, y_predict))
        print("DATA SHAPE: ", evaluation_data.shape)
        np.savetxt(out_file, evaluation_data, fmt='%.7f', header='x,y,z', delimiter=',', comments='')

    x_predict, evaluation_data = load_ds(eval_path, scaler_path)
    band_names = get_band_names(eval_path)
    print("Band names: ", band_names)

    print("Running model to get predictions for "+ eval_path)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    predict(x_predict, evaluation_data, out_dir+tmp_pred, model_path)
    ds = gdal.Open(eval_path)
    gt = ds.GetGeoTransform()
    print("Running rasterize...")
    rasterize(out_dir+tmp_pred, predictions, gt[1], gt[5])
    os.remove(out_dir+tmp_pred)

