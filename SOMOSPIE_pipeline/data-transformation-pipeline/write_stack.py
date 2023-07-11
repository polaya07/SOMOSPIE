def write_stack(vrt_file:str, out_file:str)->str:
    from osgeo import gdal
    
    translate_options = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'], callback=gdal.TermProgress_nocb)
    gdal.Translate(out_file, vrt_file, options=translate_options)
    return out_file