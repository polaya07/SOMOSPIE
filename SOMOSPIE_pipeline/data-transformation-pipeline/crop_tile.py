def crop_tile(raster:str, out_file:str, n_tiles:int, idx_x:int, idx_y:int)->str:
     # Packages
    from osgeo import gdal
    import numpy as np
    import math


    # idx_x number of the tile in the x dimension
    ds = gdal.Open(raster, 0)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    x_win_size = int(math.ceil(cols / n_tiles))
    y_win_size = int(math.ceil(rows / n_tiles))
    
    idx_x = range(0, cols, x_win_size)[idx_x]
    idx_y = range(0, rows, y_win_size)[idx_y]

    if idx_y + y_win_size < rows:
        nrows = y_win_size
    else:
        nrows = rows - idx_y

    if idx_x + x_win_size < cols:
        ncols = x_win_size
    else:
        ncols = cols - idx_x

    translate_options = gdal.TranslateOptions(srcWin=[idx_x, idx_y, ncols, nrows], creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'], callback=gdal.TermProgress_nocb)
    gdal.Translate(out_file, raster, options=translate_options)
    return out_file
