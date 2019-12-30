#!/bin/bash
# some good documentation on SLURM jobs: https://hpcc.usc.edu/support/documentation/slurm/
# '#SBATCH --[var]=[var_value]' defines parameters for running on the cluster

#******* MODIFY BETWEEN JOBS ******
#SBATCH --job-name=jobs_echo
#**********************************

#*** Number of cores to run on ****
# Note that 1 core is needed as a controller for worker cores
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --nodes=8
#**********************************

#******* Wall clock limit *********
# Hours : Minutes : Seconds (e.g., 100 hours = 4 days, 4 hours)
#SBATCH --time=72:00:00
#**********************************

# Run '>>>sacctmgr -p show associations user=$USER' to find
#  account_name and partition_name.
# See 'http://research-it.berkeley.edu/services/high-performance-computing/running-your-jobs#Key-options'
#  for details.
#SBATCH --account=fc_ocow
#SBATCH --partition=savio2

# for interactive:
#   srun --pty -A fc_ocow --partition=savio2 --qos=savio_normal --ntasks=1 --nodes=1 --ntasks-per-node=24 --time=72:00:00 bash -i
JOBSJSON=$1

ECHO_DIR=$(pwd)
BRC_DIR=${ECHO_DIR}/brc
if [[ $ECHO_DIR != */echo ]]; then
  echo -e "${RED} Please run this script from the 'echo' directory ${NC}";
  return 0 2> /dev/null || exit 0
fi;

module load python/3.6 gcc openmpi

if [ $SLURM_JOB_NUM_NODES = 1 ]; then
  ipcluster start -n $SLURM_NTASKS &
  sleep 30 # wait until all engines have successfully started
else
  echo running this one
  ipcontroller --ip='*' --nodb &
  sleep 90
  srun ipengine &
  sleep 180
fi

echo begin execution
time ipython $BRC_DIR/run_experiment_ipyparallel.py -- --jobs-json=$JOBSJSON
#ipcluster stop
exit
