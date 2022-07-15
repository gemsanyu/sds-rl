from copy import deepcopy
import json
import math
import pathlib

import multiprocessing as mp
from random import randint, random

NUM_RES = 128
FIELDS =['id', 'res', 'subtime', 'walltime', 'profile', 'user_id']
PROFILE_DICT = {"cpu": 100000000, "com": 0, "type": "parallel_homogeneous"}
BASE_PROFILE_DICT = {"100": PROFILE_DICT}

# the file to be converted to 
# json format
filename_nasa = 'raw/nasa.txt'

# open file
def get_job_list(filepath, num_of_res=NUM_RES):
    max_res = 0
    with open(filepath.absolute()) as fh:
        job_list = []

        # read the file and turn all of it into list
        for line in fh:
                # reading line by line from the text file
                description = list(line.strip().split(None, 18))
        
                # intermediate dictionary
                job = {"id": 0, "res": 0, "subtime": 0, "walltime": 0, "profile": "100", "user_id": 0}
                job["id"] = int(description[0])
                job["subtime"] = int(description[1])

                if int(description[4]) <= 0:
                    job["res"] = 1
                else:
                    job["res"] = int(description[4])
                
                if int(description[3]) <= 0:
                    continue
                else:
                    job["walltime"] = int(description[3])

                if job["res"] > max_res:
                    max_res = job["res"]
                
                job_list.append(job)
        
        # normalize job res in the list
        for job in job_list:
            job["res"] = math.ceil((job["res"]/max_res) * num_of_res)
    return job_list

def generate_dataset(dataset_idx, job_list, max_dataset_len=1000, random_discard=False):
    # 1 dataset = 1000
    # start_idx = rand(0,18239-1000)
    # idx = range(start_idx,start_idx+num_of_jobs)

    # iterate through the list based on size_of_dataset (eg. 1000), normalize the subtime, and turn it into a file
    start_idx = randint(0, len(job_list)-max_dataset_len-1)
    dataset_job_list = []
    base_subtime = job_list[start_idx]["subtime"]

    for i in range(start_idx, start_idx+max_dataset_len):
        job = deepcopy(job_list[i])
        job["subtime"] -= base_subtime
        if random_discard and random()>0.5:
            continue
        dataset_job_list += [job]
    workloads = {"nb_res": NUM_RES, "jobs": dataset_job_list, "profiles": BASE_PROFILE_DICT}
    dataset_root = "dataset"
    dataset_dir = pathlib.Path(".")/dataset_root
    dataset_dir.mkdir(parents=True, exist_ok=True)
    filename = "dataset-"+str(dataset_idx)+".json"
    filepath = dataset_dir/filename
    with open(filepath.absolute(), "w") as out_file:
        json.dump(workloads, out_file, indent = 4)

if __name__ == "__main__":
    raw_file_dir = pathlib.Path(".")/"raw"
    raw_filename = "nasa.txt"
    raw_filepath = raw_file_dir/raw_filename
    job_list = get_job_list(raw_filepath, NUM_RES)
    num_dataset = 1000
    args = [(i, job_list) for i in range(num_dataset)]
    with mp.Pool(4) as pool:
        pool.starmap(generate_dataset, args)