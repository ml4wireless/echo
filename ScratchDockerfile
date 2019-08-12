#The following makes a 1.89GB image without jupyter notebook 2.35GB with:
FROM continuumio/miniconda3
ARG CONDA_DIR=/opt/conda
ENV PATH $CONDA_DIR/bin:$PATH
RUN conda install --yes --quiet -c conda-forge jsonnet
RUN conda install --yes --quiet nomkl
RUN conda install --yes --quiet -c conda-forge pytorch-cpu
RUN conda install --yes --quiet --freeze-installed matplotlib
RUN conda install --yes --quiet --freeze-installed jupyter \
    && mkdir /opt/notebooks
RUN conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete


#The following creates a 2.45 GB image:
# FROM ufoym/deepo:pytorch-py36-cpu
# USER root
# RUN pip install --no-cache-dir jsonnet

COPY . /echo
