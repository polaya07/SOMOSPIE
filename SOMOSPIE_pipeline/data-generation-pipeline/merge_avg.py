def merge_avg(input_files:str, output_file:str)->str:
    import os
    import glob
    from osgeo import gdal # Install in a conda env: https://anaconda.org/conda-forge/gdal
    import subprocess

    # Import function
    def bash(argv):
        arg_seq = [str(arg) for arg in argv]
        proc = subprocess.Popen(arg_seq)#, shell=True)
        proc.wait() #... unless intentionally asynchronous

    vrt = gdal.BuildVRT('merged.vrt', input_files)
    vrt = None  # closes file

    with open('merged.vrt', 'r') as f:
        contents = f.read()

    if '<NoDataValue>' in contents:
        nodata_value = contents[contents.index('<NoDataValue>') + len('<NoDataValue>'): contents.index('</NoDataValue>')]# To add averaging function
    else:
        nodata_value = 0

    code = '''band="1" subClass="VRTDerivedRasterBand">
  <PixelFunctionType>average</PixelFunctionType>
  <PixelFunctionLanguage>Python</PixelFunctionLanguage>
  <PixelFunctionCode><![CDATA[
import numpy as np

def average(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    data = np.ma.array(in_ar, mask=np.equal(in_ar, {}))
    np.mean(data, axis=0, out=out_ar, dtype="float32")
    mask = np.all(data.mask,axis = 0)
    out_ar[mask] = {}
]]>
  </PixelFunctionCode>'''.format(nodata_value, nodata_value)

    sub1, sub2 = contents.split('band="1">', 1)
    contents = sub1 + code + sub2

    with open('merged.vrt', 'w') as f:
        f.write(contents)

    cmd = ['gdal_translate', '-co', 'COMPRESS=LZW', '-co', 'TILED=YES', '-co', 'BIGTIFF=YES', '--config', 'GDAL_VRT_ENABLE_PYTHON', 'YES', 'merged.vrt', output_file]
    bash(cmd)
    os.remove('merged.vrt')
    return output_file
