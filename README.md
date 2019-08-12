# echo

Lost? Try running: 

```./runecho```

When prompted, 
- you are not on the BRC
- you do not have a jobs.json
- you want to run a script: `scripts/single`
- yes it should have made and returned the corrent jobs.json file
- run one job for job id = 0

# Setup 

Please install a virtual env manager, Anaconda is suggested with Python 3. Create a new environment and set up there!
```
> conda create -n echo_env python=3.6
> source activate echo_env
> pip install --user --requirement requirements.txt
> ./runecho
```

(in progress) A Dockerfile is also supplied. With the current Dockerfile, you can run jupyter notebook. Please edit it to fit your usage. 
```
> docker build . -t ECHO
> docker run -it -p 8888:8888 ECHO /bin/bash -c "jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser --allow-root"
```
Go to localhost:8888 and copy and paste the token over.

Don't forget to shutdown your containers.
```
> docker ps -a 
> docker rm [NAME]
```
Here's a useful command to rm all containers that are Exited (usually because of an error...), if you, like me are bad with Docker: ` docker rm $(docker ps -a | grep Exited | awk '{print $1}')`

# Terminology

**protocol** - the information protocol (i.e. `gradient_passing`,`loss_passing`,`shared_preamble`,`private_preamble`) 

**models** - the model used to learn with (`classic`, `neural`, `poly`)

**mod_order** - `QPSK`,`8PSK`,`QAM16` for 2, 3, or 4 bits per symbol

**experiment** - protocol + mod_order + agent model(s) 
Examples: `experiments/gradient_passing/QPSK_neural_and_classic/` or `experiments/shared_preamble/QPSK_neural_vs_clone/`

**trial** - an instantiation of a single experiment via sampling and generating new seeds

**job** - same as trial but agnostic to the experiment because you can have trials of multiple experiments in one jobs.json file

# Code Overview

`./experiments/[protocol]/model_params.libsonnet` stores the default model parameters for running an experiment under that protocol

`./experiments/create_experiment.py` is used to create an experiment with default settings or command-line specified settings. (See examples of usage in `./scripts/single` and `./scripts/singlecustom`.)

`./experiments/make_jobs.py` is used to create a jobs.json file containing the parameters to run multiple trials (sampling or generating seeds) for a single experiment. *If you supply a file with multiple experiments (one per line), a single jobs.json will be made containing jobs for all of the experiments listed.* (See examples of usage in `./scripts/single`, `./scripts/singlecustom`, `./utils/preprocess_experiments.py`.)

`./utils/preprocess_experiments.py` creates and makes jobs for ALL of the default experiments. Or for a specific protocol. (See examples of usage in `./scripts/singleprotocol` and `./scripts/all`.)

`./run_experiment.py` is used to run experiments from a jobs.json file.

`./plot_experiment.py` is used to plot the results of a single job output.

# Useful commands
`./utils/clean` : cleans out results, experiments and temporary files. 

`./utils/clrbrc` : cleans out outputs from running on brc

`./scripts/single` : see the file. creates 1 experiment using default settings, makes the jobs, and then runs the first job

`./scripts/singlecustom` : see the file. creates 1 experiment using custom params in the custom_params folder, makes the jobs, and then runs the first job

`./scripts/singleprotocol`: see the file, creates the default experiments for a single protocol, makes jobs into a single jobs.json file, runs the first job.

`./scripts/all`: see the file, creates the default experiments for all protocols, makes jobs into a single jobs.json file, runs the first job.

`./runecho`: command-line helper for running experiments. 

# Running on BRC

`./runecho`: Use this tool to help you. It knows what to do. 
`./utils/clrbrc`: Use this tool to clean up after you've finished up running to clean up intermediate files

 Useful commands:
* `squeue -u  caryntran`: Jobs on the queue / being run
* `scancel [job-id]`: Cancel a job
* `sinfo -p savio2`: See nodes available and being used
* `wwall -j [job-id]`: See the utilization of the nodes
 
# Running on Google Cloud Platform

Resources:
 * [https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction]
 * **[https://cloud.google.com/ml-engine/docs/custom-containers-training]**
 * **[https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/pytorch/containers/hp_tuning]**
 * [https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/pytorch]
 * [https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-keras]
 
## Prerequisites
1. Make GCloud Account
2. Install cloud sdk, docker 
3. Make a project + enable billing
4. Make a bucket 
5. Enable APIs: AI Platform ("Cloud Machine Learning Engine"), Compute Engine, and Google Container Registry API

## Setup
Set these environment variables to make your life easy. Replace bracketed values with UNIQUE identifiers.
These must correspond to the project_id and bucket_name you created on GCloud console.
``` 
export PROJECT_ID=[torch-echo]
export BUCKET_NAME=[sahai_echo]
```
These can be whatever you want:
```
export BASE_NAME=[echo_tuning]
export IMAGE_REPO_NAME=$BASE_NAME_pytorch_container
export IMAGE_TAG=$BASE_NAME_pytorch

export NOW=$(date +%Y%m%d_%H%M%S)
export JOB_NAME=$BASE_NAME_container_job_${NOW}
export MODEL_DIR=$BASE_NAME_pytorch_model_${NOW}
export JOB_DIR=gs://$BUCKET_NAME/$MODEL_DIR
export REGION=us-west1
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
```
# Prepare the container
Build the Dockerfile
Test it locally
Push it onto Google Cloud
```
docker build -f Dockerfile -t $IMAGE_URI ./
docker run $IMAGE_URI --total-batches 10 --log-interval 2 
docker push $IMAGE_URI
```
# Submit job
Run on Google Cloud
```
gcloud beta ai-platform jobs submit training $JOB_NAME \
  --job-dir=$JOB_DIR \
  --region=$REGION \
  --master-image-uri $IMAGE_URI \
  --config=config.yaml \
  --scale-tier BASIC
  ```
 # Customize
 `config.yaml` and `trainer/task.py` will be of interest to you
