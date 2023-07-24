
def crop_region_eval(input_file:str, shp_file:str, output_file:str)->str:
    from osgeo import gdal

    warp_options = gdal.WarpOptions(cutlineDSName=shp_file, cropToCutline=True, creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'], callback=gdal.TermProgress_nocb)
    warp = gdal.Warp(output_file, input_file, options=warp_options)
    warp = None
    return output_file

def crop_tile(raster:str, out_file:str, n_tiles:int, idx_x:int, idx_y:int)->str:
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
    gdal.Translate(out_file, raster, options=translate_options)_size < rows:
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

def build_stack_eval(input_files:list, vrt_file:str)->str:

    from osgeo import gdal, ogr  # Install in a conda env: https://anaconda.org/conda-forge/gdal
    
    # input_files: list of .tif files to stack
    vrt_options = gdal.BuildVRTOptions(separate=True)
    vrt = gdal.BuildVRT(vrt_file, input_files, options=vrt_options)
    #translate_options = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'],
    #                                          callback=gdal.TermProgress_nocb)
    #gdal.Translate(output_file, vrt, options=translate_options)
    vrt = None  # closes file
    return vrt_file

def write_stack(vrt_file:str, out_file:str)->str:
    from osgeo import gdal
    
    translate_options = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'], callback=gdal.TermProgress_nocb)
    gdal.Translate(out_file, vrt_file, options=translate_options)
    return out_file

def band_names(output_file:str, band_names:list, vrt_file:str):
    
    def get_band_names(raster):
        ds = gdal.Open(raster, 0)
        names = []
        for band in range(ds.RasterCount):
            b = s.GetRasterBand(band + 1)
            names.append(b.GetDescription())
        ds = None
        return names

    def set_band_names(raster, band_names):
        ds = gdal.Open(raster, 0)
        print(ds.RasterCount)
        for i, name in enumerate(band_names):
            b = ds.GetRasterBand(i + 1)
            b.SetDescription(name)
        del ds
    
    set_band_names(output_file, parameter_names)
    os.remove(vrt_file)
    print("Band names:")
    print(get_band_names(output_file))


