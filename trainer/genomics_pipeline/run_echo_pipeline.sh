#!/bin/bash
#YOU MUST FIRST CREATE ALL THE JOBS.JSON INTO THE trainer/work folder!!!!
#first run echo/scripts/all
#then run process for pipeline
#you should move all these files in genomics_pipeline to the trainer directory
# ALSO DON'T FORGET ABOUT YOUR  datalab connect --zone us-west1-c --port 8081 echo-datalab

BUCKET_NAME=torch-echo
NOW=$(date +%Y%m%d_%H%M%S)
JOB_NAME=echo_${NOW}
JOB_DIR="gs://${BUCKET_NAME}/${JOB_NAME}"

ECHO_DIR=$(pwd)
TRAINER_DIR=${ECHO_DIR}/trainer

if [[ $ECHO_DIR != */echo ]]; then
  echo -e "${RED} Please run this script from the 'echo' directory ${NC}";
  return 0 2> /dev/null || exit 0
fi;

# spinning up tasks
echo "spinning up tasks"

#rm -f ${TRAINER_DIR}/genomics_pipeline/operations
touch ${TRAINER_DIR}/genomics_pipeline/operations_${NOW}

#rm -f ${TRAINER_DIR}/genomics_pipeline/operations_keys
touch ${TRAINER_DIR}/genomics_pipeline/operations_keys_${NOW}

TO_RUN=("clone_0" "clone_1" "neural_0" "neural_1")
#for y in {0..7}; do
#TO_RUN=( 1179 1476 )
for y in "${TO_RUN[@]}"; do
  for x in {0..50}; do
      echo -n "."
      gcloud alpha genomics pipelines run \
      --pipeline-file ${TRAINER_DIR}/genomics_pipeline/echo-pipeline.yaml \
      --logging ${JOB_DIR}/logs/task${x}_${y}.log \
      --inputs TASK_ID=$x,EXP_ID=$y,JOB_DIR=$JOB_DIR  \
      --preemptible \
      >> ${TRAINER_DIR}/genomics_pipeline/operations_${NOW} 2>&1
      echo "($x $y)" >> ${TRAINER_DIR}/genomics_pipeline/operations_keys_${NOW}
  done
done

#echo -e "\ncomplete"
#

#echo "waiting for all jobs to complete"

#for op in `cat ${TRAINER_DIR}/operations | cut -d / -f 2 | cut -d ']' -f 1`; do
#      echo -n "."
#      CMD="gcloud --format='value(done)' alpha genomics operations describe $op"
#      while [[ $(eval ${CMD}) != "True" ]]; do echo -n "X"; sleep 5; done
#done
#
#echo -e "\nall jobs done"
#
#gsutil ls ${JOB_DIR}

