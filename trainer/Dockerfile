FROM continuumio/miniconda3

WORKDIR /root
ARG CONDA_DIR=/opt/conda
ENV PATH $CONDA_DIR/bin:$PATH
RUN conda install --yes --quiet -c conda-forge pytorch-cpu
RUN pip install google-cloud-storage
RUN pip install cloudml-hypertune


# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg


COPY . /root/echo
WORKDIR /root/echo

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "./trainer/task_hyperparam_search.py"]
