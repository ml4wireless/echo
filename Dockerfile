FROM continuumio/miniconda3

WORKDIR /root
ARG CONDA_DIR=/opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

########### OPTION FOR RUNNING LOCALLY WITH JUPYTER NOTEBOOK -- 2.35GB (1.89GB without jupyter notebook)
#
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
#
###########

########### OPTION FOR PACKAGING AS A CUSTOM CONTAINER FOR GCLOUD AI PLATFORM HYPERTUNE
#
#    RUN conda install --yes --quiet -c conda-forge pytorch-cpu
#    RUN pip install google-cloud-storage
#    RUN pip install cloudml-hypertune
#    # Path configuration
#    ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
#    # Make sure gsutil will use the default service account
#    RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg
#    #Sets up the entry point to invoke the trainer.
#    ENTRYPOINT ["python", "./trainer/task_hyperparam_search.py"]
#
###########

COPY . /root/echo
WORKDIR /root/echo
RUN pwd


