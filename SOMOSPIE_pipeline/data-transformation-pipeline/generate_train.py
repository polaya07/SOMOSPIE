#!/usr/bin/env python3


def build_stack_train(dir: str,satellite_file: str, input_files: list, output_file:str, year:int, month:int)->str:

    from osgeo import gdal
    import os

    # input_files: list of .tif files to stack
    input_files.insert(0, satellite_file)
    
    # Get target resolution from satellite file
    ds = gdal.Open(input_files[0], 0)
    xmin, xres, _, ymax, _, yres = ds.GetGeoTransform()
    for i in input_files:
        rds=gdal.Open(i)

    vrt_file = dir+'{0:04d}_{1:02d}_stack.vrt'.format(year, month)
    vrt_options = gdal.BuildVRTOptions(separate=True)
    vrt = gdal.BuildVRT(vrt_file, input_files, options=vrt_options)
    translate_options = gdal.TranslateOptions(creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'], xRes=xres , yRes=yres,
                                              callback=gdal.TermProgress_nocb)
    gdal.Translate(output_file, vrt, options=translate_options)
    vrt = None  # closes file
    os.remove(vrt_file)
    return output_file

def get_shp(zip_file: str, dir:str)->str:
    import zipfile
    import os
    from pathlib import Path

    Path(dir+'shp_file').mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dir+zip_file, 'r') as zip_ref:
        zip_ref.extractall(dir+'shp_file')

    shp_files=[]
    for root, dirs, files in os.walk(dir+'shp_file'):
        for file in files:
            if file.endswith(".shp"):
                shp_files.append(os.path.join(root, file))
    shp_file=shp_files[0]
    print(shp_file)
    return shp_file

def crop_region_train(input_file:str, shp_file:str, output_file:str, parameter_names:list, year:int, month:int):
    from osgeo import gdal
    from pathlib import Path

    def get_band_names(raster):
        ds = gdal.Open(raster, 0)
        names = []
        for band in range(ds.RasterCount):
            b = ds.GetRasterBand(band + 1)
            names.append(b.GetDescription())
        ds = None
        return names

    def set_band_names(raster, band_names):
        ds = gdal.Open(raster, 0)
        for i, name in enumerate(band_names):
            b = ds.GetRasterBand(i + 1)
            b.SetDescription(name)
        ds = None

    parameter_names.insert(0, 'z')
    print(parameter_names)

    warp_options = gdal.WarpOptions(cutlineDSName=shp_file, cropToCutline=True, creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'],
                                    callback=gdal.TermProgress_nocb)
    warp = gdal.Warp(output_file, input_file, options=warp_options)
    warp = None

    set_band_names(output_file, parameter_names)
    print(get_band_names(output_file))
    
    
    