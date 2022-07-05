import json
import math
import pathlib

num_of_res = 128

fields =['id', 'res', 'subtime', 'walltime', 'profile', 'user_id']

p_dict = {"cpu": 100000000, "com": 0, "type": "parallel_homogeneous"}
base_profile_dict = {"100": p_dict}

# the file to be converted to 
# json format
filename_nasa = 'raw/nasa.txt'
filename_gaia = 'raw/gaia.txt'
filename_llnl = 'raw/llnl.txt'

dataset_root = "dataset"
dataset_dir = pathlib.Path(".")/dataset_root
dataset_dir.mkdir(parents=True, exist_ok=True)

max_res = 0


size_of_dataset = 1000
counter = 1

# foreach file:
#   for each line in file text:
#       append to job_list
#   normalize the job_list res
#   foreach iteration in range (math.ceil(num_of_jobs/size_of_dataset)):
#       normalize subtime
#       create into dataset-{{counter}}.json
  
# open file
with open(filename_nasa) as fh:
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

    num_of_jobs = 18239
    # 1 dataset = 1000
    # start_idx = rand(0,18239-1000)
    # idx = range(start_idx,start_idx+num_of_jobs)

    # iterate through the list based on size_of_dataset (eg. 1000), normalize the subtime, and turn it into a file
    for iteration in range(math.ceil(num_of_jobs/size_of_dataset)):
        base_subtime = 0
        iteration_job_list = []

        # normalizing subtime
        for i in range(size_of_dataset):
            index_job_list = iteration * size_of_dataset + i

            if index_job_list >= len(job_list):
                break

            if index_job_list == iteration * size_of_dataset:
                base_subtime = job_list[index_job_list]["subtime"]
            
            job_list[index_job_list]["subtime"] = job_list[index_job_list]["subtime"] - base_subtime
            
            iteration_job_list.append(job_list[index_job_list])

        workloads = {"nb_res": num_of_res, "jobs": iteration_job_list, "profiles": base_profile_dict}
        # creating json file
        filename = "dataset-"+str(counter)+".json"
        filepath = dataset_dir/filename
        out_file = open(filepath.absolute(), "w")
        json.dump(workloads, out_file, indent = 4)
        out_file.close()

        counter += 1

#GAIA
with open(filename_gaia) as fh:
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

    num_of_jobs = 51987
    # 1 dataset = 1000
    # start_idx = rand(0,18239-1000)
    # idx = range(start_idx,start_idx+num_of_jobs)

    # iterate through the list based on size_of_dataset (eg. 1000), normalize the subtime, and turn it into a file
    for iteration in range(math.ceil(num_of_jobs/size_of_dataset)):
        base_subtime = 0
        iteration_job_list = []

        # normalizing subtime
        for i in range(size_of_dataset):
            index_job_list = iteration * size_of_dataset + i

            if index_job_list >= len(job_list):
                break

            if index_job_list == iteration * size_of_dataset:
                base_subtime = job_list[index_job_list]["subtime"]
            
            job_list[index_job_list]["subtime"] = job_list[index_job_list]["subtime"] - base_subtime
            
            iteration_job_list.append(job_list[index_job_list])

        workloads = {"nb_res": num_of_res, "jobs": iteration_job_list, "profiles": base_profile_dict}
        # creating json file
        filename = "dataset-"+str(counter)+".json"
        filepath = dataset_dir/filename
        out_file = open(filepath.absolute(), "w")
        json.dump(workloads, out_file, indent = 4)
        out_file.close()

        counter+=1


#GAIA
with open(filename_llnl) as fh:
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

    num_of_jobs = 121039
    # 1 dataset = 1000
    # start_idx = rand(0,18239-1000)
    # idx = range(start_idx,start_idx+num_of_jobs)

    # iterate through the list based on size_of_dataset (eg. 1000), normalize the subtime, and turn it into a file
    for iteration in range(math.ceil(num_of_jobs/size_of_dataset)):
        base_subtime = 0
        iteration_job_list = []

        # normalizing subtime
        for i in range(size_of_dataset):
            index_job_list = iteration * size_of_dataset + i

            if index_job_list >= len(job_list):
                break

            if index_job_list == iteration * size_of_dataset:
                base_subtime = job_list[index_job_list]["subtime"]
            
            job_list[index_job_list]["subtime"] = job_list[index_job_list]["subtime"] - base_subtime
            
            iteration_job_list.append(job_list[index_job_list])

        workloads = {"nb_res": num_of_res, "jobs": iteration_job_list, "profiles": base_profile_dict}
        # creating json file
        filename = "dataset-"+str(counter)+".json"
        filepath = dataset_dir/filename
        out_file = open(filepath.absolute(), "w")
        json.dump(workloads, out_file, indent = 4)
        out_file.close()

        counter+=1