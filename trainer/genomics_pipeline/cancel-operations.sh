#!/bin/bash

ECHO_DIR=$(pwd)
TRAINER_DIR=${ECHO_DIR}/trainer

if [[ $ECHO_DIR != */echo ]]; then
  echo -e "${RED} Please run this script from the 'echo' directory ${NC}";
  return 0 2> /dev/null || exit 0
fi;


for op in `cat ${TRAINER_DIR}/genomics_pipeline/operations_20190815_135051 | cut -d / -f 2 | cut -d ']' -f 1`; do
#      echo -n "."
      CMD="gcloud --format='value(done)' alpha genomics operations describe $op"
      if [[ $(eval ${CMD}) != "True" ]]; then
            echo $op
            CANCEL="gcloud alpha genomics operations cancel $op"
            $(eval yes Y| ${CANCEL})
      fi
#      do echo -n "X"; sleep 5; done
done