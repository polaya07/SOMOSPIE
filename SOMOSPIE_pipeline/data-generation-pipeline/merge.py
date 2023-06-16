#!/usr/bin/python3

def merge_tiles(input_dir:str, output_file:str, cos:str)->str:
    # Packages
    import os
    import glob
    from osgeo import gdal
    input_files=[]
    # get all files from dems directory
    for root, dirs, files in os.walk(os.path.abspath(input_dir)):
        for file in files:
            input_files.append(os.path.join(root, file))
    #input_files=os.listdir(input_dir)
    print("Reading TIFFs from: ", input_files)
    # input_files: list of .tif files to merge
    vrt = gdal.BuildVRT(cos+'merged.vrt', input_files)
    translate_options = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'NUM_THREADS=ALL_CPUS'], callback=gdal.TermProgress_nocb)
    gdal.Translate(output_file, vrt, options=translate_options)
    vrt = None  # closes file
    os.remove(cos+'merged.vrt')
    return output_file