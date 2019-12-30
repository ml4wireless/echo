import getpass
import json
import os
import sys
from datetime import datetime

BRC_DIR = os.path.dirname(os.path.realpath(__file__))
ECHO_DIR = os.path.dirname(BRC_DIR)

assert len(sys.argv) == 3, "make_tasks.py requires 2 positional arguments: [location of jobs json] [number of nodes]"
jobs_json = sys.argv[1]
number_nodes = int(sys.argv[2])

with open(jobs_json, "r") as jf:
    jobs = json.load(jf)

# while os.path.exists(BRC_DIR+"/out/taskfile%s"%suff):
#     i += 1
#     suffix = str(i)


assert os.path.isdir('/global/scratch/'), "ARE YOU SURE YOU ARE LOGGED INTO THE BRC?"
echo_symlink_dir = '/global/scratch/%s/echo/' % (getpass.getuser())
os.makedirs(echo_symlink_dir, exist_ok=True)
assert os.path.isdir(echo_symlink_dir)
now = datetime.now()  # current date and time
job_date = now.strftime("%y%m%d%H%M%S")

total_num_jobs = len(jobs)
batch_sizes = [total_num_jobs // number_nodes] * number_nodes
left_over = total_num_jobs - (total_num_jobs // number_nodes) * number_nodes
for l in range(left_over):
    batch_sizes[l] += 1
BRC_OUT = BRC_DIR + "/out"
for j in range(number_nodes):
    suffix = len([item for item in os.listdir(BRC_OUT) if 'taskfile' in item])
    start = sum(batch_sizes[:j])
    end = start + batch_sizes[j]  # -1
    node_jobs = jobs[start:end]
    node_jobs_json = BRC_DIR + "/out" + "/jobs%i.json" % suffix
    node_task_file = BRC_OUT + "/taskfile%s" % suffix

    with open(node_jobs_json, "w") as njf:
        njf.write(json.dumps(node_jobs, indent=4, sort_keys=False))

    with open(node_task_file, "w") as tf:
        tf.write(
            "python %s/run_experiment.py --jobs_file=%s --job_id=$HT_TASK_ID --echo_symlink_to=%s --job_date=%s\n" % (
                ECHO_DIR, node_jobs_json, echo_symlink_dir, job_date))

    print(node_task_file)
    print(batch_sizes[j])
