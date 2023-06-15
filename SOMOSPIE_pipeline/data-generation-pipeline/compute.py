def compute_geotiled(input_file:str, aspect_file:str, hillshading_file:str, slope_file:str) -> -> NamedTuple('Output', [("aspect", str), 
                                                                                                                        ('hill', str),
                                                                                                                        ('slope', str)]):
    ## Packages
    import os
    from osgeo import gdal

    # Slope
    dem_options = gdal.DEMProcessingOptions(format='GTiff', creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'])
    gdal.DEMProcessing(slope_file, input_file, processing='slope', options=dem_options)
    # Aspect
    dem_options = gdal.DEMProcessingOptions(zeroForFlat=True, format='GTiff', creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'])
    gdal.DEMProcessing(aspect_file, input_file, processing='aspect', options=dem_options)
    # Hillshading
    dem_options = gdal.DEMProcessingOptions(format='GTiff', creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'])
    gdal.DEMProcessing(hillshading_file, input_file, processing='hillshade', options=dem_options)

    return(aspect_file,hillshading_file,slope_file)
