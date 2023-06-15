def merge_tiles(input_files:str, output_file:str)->str:
    # Packages
    import os
    import glob
    from osgeo import gdal
    # input_files: list of .tif files to merge
    vrt = gdal.BuildVRT('merged.vrt', input_files)
    translate_options = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'NUM_THREADS=ALL_CPUS'], callback=gdal.TermProgress_nocb)
    gdal.Translate(output_file, vrt, options=translate_options)
    vrt = None  # closes file
    os.remove('merged.vrt')
    return output_file