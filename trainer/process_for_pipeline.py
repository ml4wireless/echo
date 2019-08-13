import csv, os, json


def rm_mkdir(dir):
    if os.path.isdir(dir):
        import shutil
        shutil.rmtree(dir)
    os.makedirs(dir)
    return


JOB_DIR = os.environ['JOB_DIR']
TRAINER_DIR = os.path.dirname(os.path.realpath(__file__))
ECHO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
WORK_DIR = os.path.join(TRAINER_DIR, 'work')

rm_mkdir(WORK_DIR)
print(ECHO_DIR)

with open('%s/experiments/jobs.json' % ECHO_DIR) as jfile:
    jobs = json.load(jfile)

NUM_JOBS = len(jobs)

for i, job in enumerate(jobs):
    with open('%s/%i.json' % (WORK_DIR, i), 'w') as jf:
        json.dump(job, jf)

with open('%s/pipeline-tasks.tsv' % TRAINER_DIR, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['--env INDEX', '--output RESULT_FILE'])
    for i in range(NUM_JOBS):
        tsv_writer.writerow(["%i" % i, '%s/%i.npy' % (JOB_DIR, i)])
