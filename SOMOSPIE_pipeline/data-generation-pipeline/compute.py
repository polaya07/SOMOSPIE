from typing import NamedTuple

def compute_geotiled(dir: str, tile_count:int, input_file:str, aspect_file:str, hillshading_file:str, slope_file:str) -> NamedTuple('Output', [("aspect", str), 
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
    gdal.DEMProcessing(dir+str(tile_count)+'hill.tif', input_file, processing='hillshade', options=dem_options)

    # Change datatype of hillshading to the same as the other parameters
    translate_options = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'NUM_THREADS=ALL_CPUS'], outputType=gdal.GDT_Float32, callback=gdal.TermProgress_nocb)
    gdal.Translate(hillshading_file, dir+str(tile_count)+'hill.tif', options=translate_options)
    os.remove(dir+str(tile_count)+'hill.tif')

    return(aspect_file,hillshading_file,slope_file)
