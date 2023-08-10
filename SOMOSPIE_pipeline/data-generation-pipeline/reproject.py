#!/usr/bin/python3

def reproject( input_file: str, output_file:str, projection: str, nodata:str='n', pipeline_flag: list=[])->str:
    # Packages
    import os
    from osgeo import gdal
    import numpy as np 

    if os.path.isfile(output_file):
        return output_file
    else:  
        # Projection can be EPSG:4326, .... or the path to a wkt file
        if nodata == 'y':
            warp_options = gdal.WarpOptions(dstSRS=projection, dstNodata=np.nan, creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'NUM_THREADS=ALL_CPUS'], multithread=True, warpOptions=['NUM_THREADS=ALL_CPUS'], callback=gdal.TermProgress_nocb)
        else:
            warp_options = gdal.WarpOptions(dstSRS=projection, creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'NUM_THREADS=ALL_CPUS'], callback=gdal.TermProgress_nocb, multithread=True, warpOptions=['NUM_THREADS=ALL_CPUS'])
        warp = gdal.Warp(output_file, input_file, options=warp_options)
        warp = None  # Closes the files
        return output_file
