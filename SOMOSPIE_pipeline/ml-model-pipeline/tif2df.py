def tif2df(raster_file, band_names) :
    from osgeo import gdal, ogr  # Install in a conda env: https://anaconda.org/conda-forge/gdal
    import numpy as np
    import pandas as pd

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
    print(df.describe(include='all'))
    print("5 ENTRIES\n",df.head())
    print("Size of the df\n",df.size)
    #df.to_csv(output_file, index=None)
    return df