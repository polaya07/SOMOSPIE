#!/usr/bin/env python3

def download_parallel(year:int,dir:str)->str:
    import calendar
    import subprocess
    import shutil
    from pathlib import Path
    import numpy as np
    import os
    import multiprocessing
    import multiprocess as mp

    def bash (argv):
        arg_seq = [str(arg) for arg in argv]
        proc = subprocess.Popen(arg_seq, stdout=subprocess.PIPE, stderr=subprocess.PIPE)#, shell=True)
        proc.wait() #... unless intentionally asynchronous
        stdout, stderr = proc.communicate()

        # Error catching: https://stackoverflow.com/questions/5826427/can-a-python-script-execute-a-function-inside-a-bash-script
        if proc.returncode != 0:
            raise RuntimeError("'%s' failed, error code: '%s', stdout: '%s', stderr: '%s'" % (
                ' '.join(arg_seq), proc.returncode, stdout.rstrip(), stderr.rstrip()))
            
    def download(year, dir):
        version = 7.1 # ESA CCI version
        year_folder = './{0:04d}'.format(year)
        Path(dir+year_folder).mkdir(parents=True, exist_ok=True)

        commands = []
        for month in range(1, 13):
            for day in range(1, calendar.monthrange(year, month)[1] + 1):
                download_link = 'ftp://anon-ftp.ceda.ac.uk/neodc/esacci/soil_moisture/data/daily_files/COMBINED/v0{0:.1f}/{1:04d}/ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-{1:04d}{2:02d}{3:02d}000000-fv0{0:.1f}.nc'.format(version, year, month, day)
                #commands.append(['curl', '-C','-s', download_link, '-o', dir+'{0}/{1:02d}_{2:02d}.nc'.format(year_folder, month, day)])
                commands.append(['wget', '-N','-c', download_link, '-O', dir+'{0}/{1:02d}_{2:02d}.nc'.format(year_folder, month, day)])
                # bash(command)

        pool=mp.Pool(multiprocessing.cpu_count())
        pool.map(bash, commands)

    download(year,dir)
    return dir

def merge_avg(dir: str, year: int, month: int, output_file:str, projection: str)->str:
    from osgeo import gdal
    import glob
    import numpy as np
    import os
    import calendar
    import subprocess
    

    def bash(argv):
        arg_seq = [str(arg) for arg in argv]
        proc = subprocess.Popen(arg_seq)#, shell=True)
        proc.wait() #... unless intentionally asynchronous
        
    def reproject(input_file, output_file, projection):
        # Projection can be EPSG:4326, .... or the path to a wkt file
        warp_options = gdal.WarpOptions(dstSRS=projection, creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'NUM_THREADS=ALL_CPUS'], multithread=True, warpOptions=['NUM_THREADS=ALL_CPUS'], dstNodata=np.nan, callback=gdal.TermProgress_nocb)
        warp = gdal.Warp(output_file, input_file, options=warp_options)
        warp = None  # Closes the files

    sm_files = ['NETCDF:'+dir+'{0:04d}/{1:02d}_{2:02d}.nc:sm'.format(year, month, day) for day in range(1, calendar.monthrange(year, month)[1])]

    vrt = gdal.BuildVRT(dir+'{0:04d}_{1:02d}_merged.vrt'.format(year, month), sm_files)
    vrt = None  # closes file

    with open(dir+'{0:04d}_{1:02d}_merged.vrt'.format(year, month), 'r') as f:
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
    np.ma.mean(data, axis=0, out=out_ar, dtype="float32")
    mask = np.all(data.mask,axis = 0)
    out_ar[mask] = {}
]]>
  </PixelFunctionCode>'''.format(nodata_value, nodata_value)

    sub1, sub2 = contents.split('band="1">', 1)
    contents = sub1 + code + sub2

    with open(dir+'{0:04d}_{1:02d}_merged.vrt'.format(year, month), 'w') as f:
        f.write(contents)

    cmd = ['gdal_translate', '-co', 'COMPRESS=LZW', '-co', 'TILED=YES', '-co', 'BIGTIFF=YES', '--config', 'GDAL_VRT_ENABLE_PYTHON', 'YES', dir+'{0:04d}_{1:02d}_merged.vrt'.format(year, month), output_file]
    bash(cmd)
    os.remove(dir+'{0:04d}_{1:02d}_merged.vrt'.format(year, month))

    reproject(output_file, output_file, projection)

    return output_file




