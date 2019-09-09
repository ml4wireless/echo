import sys, os
import subprocess
from subprocess import Popen, PIPE
import time

TRAINER_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
ECHO_DIR = os.path.dirname(TRAINER_DIR)
sys.path.append(ECHO_DIR)
OPS_FILE = TRAINER_DIR + "/genomics_pipeline/operations_20190815_140031"
OPS_KEYS = TRAINER_DIR + "/genomics_pipeline/operations_keys_20190815_140031"
JOB_DIR="gs://torch-echo/echo_20190815_014749"
MAXHOURS = 3
with open(OPS_KEYS) as f: keys = [line.rstrip(')\n').lstrip("(").split(" ") for line in f]
with open(OPS_FILE) as f: ops = [line.rstrip('\].\n').lstrip('\[Running \[operations\/') for line in f]
loop = 0
start=time.time()
while False and len(ops) > 0:
    i = loop%len(ops)
    op = ops[i]
    done_command = "gcloud --format='value(done)' alpha genomics operations describe %s" % op
    err_command = "gcloud --format='value(error)' alpha genomics operations describe %s" % op
    events_command = "gcloud --flatten='metadata.events' --format='value(metadata.events.description)' " \
                  "alpha genomics operations describe %s"%op

    done_output = subprocess.check_output(done_command, shell=True)
    err_output = subprocess.check_output(err_command, shell=True)
    events_output = subprocess.check_output(err_command, shell=True)
    stopped = "stopped unexpectedly" in err_output.decode("utf-8").strip("\n")
    quota_exceeded = "Warning: Quota 'CPUS_ALL_REGIONS' exceeded." in err_output.decode("utf-8").strip("\n")
    done = done_output.decode("utf-8").strip("\n") == "True"
    if time.time() - start > MAXHOURS * 60 * 60 and not done:
        cancel_command = "gcloud alpha genomics operations cancel %s"%op
        p = Popen(cancel_command.split(" "), stdout=PIPE, stderr=PIPE, stdin=PIPE)
        p.communicate(b'y')
        key = keys.pop(i)
        op = ops.pop(i)
        loop += 1
        continue
    if quota_exceeded:
        cancel_command = "gcloud alpha genomics operations cancel %s" % op
        p = Popen(cancel_command.split(" "), stdout=PIPE, stderr=PIPE, stdin=PIPE)
        p.communicate(b'y')
        stopped = True
    if stopped:
        key = keys.pop(i)
        op = ops.pop(i)
        task_id, exp_id = key
        retry_command = "gcloud alpha genomics pipelines run --pipeline-file %s/genomics_pipeline/echo-pipeline.yaml" \
                        " --logging %s/logs/task%s_%s.log --inputs TASK_ID=%s,EXP_ID=%s,JOB_DIR=%s --preemptible" % (TRAINER_DIR, JOB_DIR, task_id, exp_id, task_id, exp_id, JOB_DIR)
        print(retry_command)
        p = Popen(retry_command.split(" "), stdin=PIPE, stdout=PIPE, stderr=PIPE)
        err, retry_output  = p.communicate(b"input data that is passed to subprocess' stdin")
        rc = p.returncode
        if rc != 0:
            continue
        new_op = retry_output.decode().rstrip('\n').rstrip('\].').lstrip('\[Running \[operations\/')
        print(new_op, task_id, exp_id)
        ops += [new_op]
        keys += [[task_id, exp_id]]
    elif done:
        keys.pop(i)
        ops.pop(i)
    loop += 1

#TO CANCEL ALL RUNNING OPS
# ops_command = "gcloud alpha genomics operations list --where=\"status = RUNNING\" --format='value(name)'"
# ops_output = subprocess.check_output(ops_command, shell=True)
# ops = [o.lstrip("operations/") for o in ops_output.decode("utf-8").split("\n")]
# print(len(ops))
# for op in ops:
#     cancel_command = "gcloud alpha genomics operations cancel %s" % op
#     p = Popen(cancel_command.split(" "), stdout=PIPE, stderr=PIPE, stdin=PIPE)
#     p.communicate(b'y')

#IDK... in the process of writing to rerun with exponential back off
# ops_command = "gcloud alpha genomics operations list --where=\"status = RUNNING\" --format='flattened(name, metadata.request.pipelineArgs.inputs)'"
# ops_output = subprocess.check_output(ops_command, shell=True)
# out = [ops_output.decode("utf-8").split("\n")]
# print(out)
