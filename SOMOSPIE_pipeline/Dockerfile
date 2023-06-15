FROM conda/miniconda3
COPY environment.yaml . 
SHELL ["/bin/bash", "--login", "-c"]
RUN conda env create -f environment.yaml 
RUN conda init bash
RUN conda activate somospie