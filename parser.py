import json
import math

num_of_res = 128

job_list = []
fields =['id', 'res', 'subtime', 'walltime', 'profile', 'user_id']

p_dict = {"cpu": 100000000, "com": 0, "type": "parallel_homogeneous"}
base_profile_dict = {"100": p_dict}

# the file to be converted to 
# json format
filename = 'raw/llnl.txt'

max_res = 0
  
# creating dictionary
with open(filename) as fh:      
    for line in fh:
          
        # reading line by line from the text file
        description = list(line.strip().split(None, 18))
          
        # for output see below
        print(description) 
  
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
        
        # appending the record of each employee to
        # the main dictionary
        job_list.append(job)

for job in job_list:
    job["res"] = math.ceil((job["res"]/max_res) * num_of_res)

workloads = {"nb_res": num_of_res, "jobs": job_list, "profiles": base_profile_dict}

# creating json file        
out_file = open("workloads-llnl-cleaned-128host.json", "w")
json.dump(workloads, out_file, indent = 4)
out_file.close()