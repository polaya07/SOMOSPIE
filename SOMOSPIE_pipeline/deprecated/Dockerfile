FROM ghcr.io/osgeo/gdal:ubuntu-small-latest
RUN apt-get -y update
RUN apt-get -y install curl
RUN apt-get -y install python3-pip 
RUN apt-get -y install --fix-missing libnetcdf-dev
RUN python3 -m pip install pandas scikit-learn 
RUN python3 -m pip install --upgrade GDAL