import csv, os, json
import string
import random
def random_generator(size=6, chars=string.ascii_uppercase):
    return ''.join(random.choice(chars) for x in range(size))

def rm_mkdir(dir):
    if os.path.isdir(dir):
        import shutil
        shutil.rmtree(dir)
    os.makedirs(dir)
    return


# JOB_DIR = os.environ['JOB_DIR']???
TRAINER_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ECHO_DIR = os.path.dirname(TRAINER_DIR)
WORK_DIR = os.path.join(TRAINER_DIR, 'work')

print(ECHO_DIR)
# os.makedirs(WORK_DIR)
jobs = []
for j, s in enumerate(['fneural','fpoly']):
    with open('%s/experiments/private_preamble/QPSK_poly_vs_clone_custom_%s/jobs.json' % (ECHO_DIR, s)) as jfile:
        jobs = json.load(jfile)

    NUM_JOBS = len(jobs)

    for i, job in enumerate(jobs):
        with open('%s/%i_poly_vs_clone_%i_%s.json' % (WORK_DIR, i, j, random_generator()), 'w') as jf:
            json.dump(job, jf)

#THIS PART IS ONLY NEEDED IF YOU ARE RUNNING A PIPELINE AND NOT JUST A BUNCH OF OPERATIONS
#FOR DSUBBBB tool... not as fast :(
# with open('%s/pipeline-tasks.tsv' % TRAINER_DIR, 'wt') as out_file:
#     tsv_writer = csv.writer(out_file, delimiter='\t')
#     tsv_writer.writerow(['--env INDEX', '--output RESULT_FILE'])
#     for i in range(NUM_JOBS):
#         tsv_writer.writerow(["%i" % i, '%s/%i.npy' % (JOB_DIR, i)])
