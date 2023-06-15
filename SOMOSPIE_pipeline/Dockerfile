FROM python:3.10-slim
RUN apt-get -y update
RUN apt-get -y install curl
COPY environment.yaml .
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate && \
    conda env create -f environment.yaml && \
    conda activate somospie