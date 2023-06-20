def build_stack(input_files:list, output_file:str)->str:

    from osgeo import gdal, ogr  # Install in a conda env: https://anaconda.org/conda-forge/gdal
    
    # input_files: list of .tif files to stack
    vrt_options = gdal.BuildVRTOptions(separate=True)
    vrt = gdal.BuildVRT("stack.vrt", input_files, options=vrt_options)
    translate_options = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'],
                                              callback=gdal.TermProgress_nocb)
    gdal.Translate(output_file, vrt, options=translate_options)
    vrt = None  # closes file
    return output_file