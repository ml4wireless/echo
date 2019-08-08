#!/bin/bash
# Job name:
#SBATCH --job-name=torch_echo
#
# Account:
#SBATCH --account=fc_ocow
#
# Partition:
#SBATCH --partition=savio2
#
# Tasks
#SBATCH --ntasks-per-node=24
#SBATCH --nodes=1
#
# Wall clock limit:
#SBATCH --time=72:00:00
#
## Command(s) to run:

#BRC_DIR=$0
TASKFILE=$1
N=$2
#folder=${TASKFILE##*/}
ECHO_DIR=$(pwd)
BRC_DIR=${ECHO_DIR}/brc
if [[ $ECHO_DIR != */echo ]]; then
  echo -e "${RED} Please run this script from the 'echo' directory ${NC}";
  return 0 2> /dev/null || exit 0
fi;

module load python/3.6 gcc openmpi
#mkdir -p "${BRC_DIR}/brc/out/${folder}"
mkdir -p "${BRC_DIR}/out/work"
ht_helper.sh -v -m "python/3.6" -t "$TASKFILE" -r "$N" -w "./brc/out/work"
