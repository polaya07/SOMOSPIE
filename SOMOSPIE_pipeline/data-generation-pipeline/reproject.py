def reproject(input_file: str, output_file:str, projection: str)->str:
    # Packages
    import os
    from osgeo import gdal

    # Projection can be EPSG:4326, .... or the path to a wkt file
    warp_options = gdal.WarpOptions(dstSRS=projection, creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'NUM_THREADS=ALL_CPUS'],
                                    callback=gdal.TermProgress_nocb, multithread=True, warpOptions=['NUM_THREADS=ALL_CPUS'])
    warp = gdal.Warp(output_file, input_file, options=warp_options)
    warp = None  # Closes the files
    return output_file
