#!/bin/bash
#YOU MUST FIRST CREATE ALL THE JOBS.JSON INTO THE trainer/work folder!!!!
#first run echo/scripts/all
#then run process for pipeline
#you should move all these files in genomics_pipeline to the trainer directory
# ALSO DON'T FORGET ABOUT YOUR  datalab connect --zone us-west1-c --port 8081 echo-datalab

BUCKET_NAME=sahai_echo
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

rm -f ${TRAINER_DIR}/operations
touch ${TRAINER_DIR}/operations

#for x in {0..2250}; do
TO_RUN=( 1179 1476 )
for x in "${TO_RUN[@]}"; do
      echo -n "."
      gcloud alpha genomics pipelines run \
      --pipeline-file ${TRAINER_DIR}/echo-pipeline.yaml \
      --logging gs://${BUCKET_NAME}/${JOB_NAME}/logs/task$x.log \
      --inputs TASK_ID=$x,JOB_DIR=$JOB_DIR  \
      --preemptible \
      >> ${TRAINER_DIR}/operations 2>&1
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

