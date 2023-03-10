# BIG-Bench Lite Evaluation of T0

## Part 1 of 4: Setting up the environments

After cloning the repo, first, modify the environment variables on lines 3, 4, 5, 8, 13, 16, 17, 18, and 19 of env_setup.sh to match your directory locations. Then, grant env_setup.sh permissions to execute, which can be done via the chmod command. With those changes made, run env_setup.sh to set up the associated Conda environment. This will create a directory system laid out below. Do not delete these files.

capstone_rst

|_ mc

|_ open



## Part 2 of 4: Running the Multiple Choice Benchmark

Run run_bench_mc.sh

This will create files for the results of the evaluation in JSON files called results.json. There will be one sub-folder in capstone_rst/mc per dataset. The directory system is laid out below. The absence of any of the results.json files indicates that the associated experiment failed. The directory system is laid out below.

capstone_rst

|_ mc
  
  |_ dataset #1
  
    |_ results.json
  
  |_ dataset #2
  
    |_ results.json
  
  ...

  |_ dataset #17

    |_ results.json

|_ open



## Part 3 of 4: Running the Free-Response Benchmark

Run run_bench_open.sh

Exact same rules as Part 2 apply here. The new directory system is laid out below.

capstone_rst

|_ mc
  
  |_ dataset #1
  
    |_ results.json
  
  |_ dataset #2
  
    |_ results.json
  
  ...

  |_ dataset #17

    |_ results.json

|_ open
  
  |_ dataset #1
  
    |_ results.json
  
  |_ dataset #2
  
    |_ results.json
  
  ...

  |_ dataset #7

    |_ results.json



## Part 4 of 4: Parsing Results

Run parse_results.py

NOTE: The current version of this file only does the following for the multiple choice results, because those are the only ones that work right now. Nothing on the user side changes from adding the free-response datasets.

This script concatenates all the experiments, the preprocessed data, the model output, and expectation into a file titled "fullResults.csv", and then aggregate figures into a file called "summary.csv". If the above directory structure is unchanged, this should run without issue.
