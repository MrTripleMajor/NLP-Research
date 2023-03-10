# Sets up pip3 and adds conda to PATH
export PATH="$PATH:/koko/system/anaconda/bin"
REPLICATION_DIR=~/Capstone/generate_all
EVAL_SCRIPT_SCORING=~/Capstone/generate_all/t0_multiQ.py
EVAL_SCRIPT_OPEN=~/Capstone/generate_all/t0_open.py

# Sets up directories for results
RESULTS_LOCATION=/common/users/ar1570
SCORING_RESULTS_DIR=${RESULTS_LOCATION}/capstone_rst/mc/
OPEN_RESULTS_DIR=${RESULTS_LOCATION}/capstone_rst/open/

# Sets up path to conda env and to BIG-Bench
CONDA_DIR=/common/users/ar1570
ENV_NAME=mypython37
PATH_TO_T0=~/Capstone/t0_replication/rte/
BIG_BENCH=/common/users/ar1570/BIG-bench

# Starts up the env
source activate ${CONDA_DIR}/${ENV_NAME}

# A function to run T0
BigBenchOpen ( ) {
    python ${EVAL_SCRIPT_OPEN} --dataset_name bigbench --dataset_config_name $1 --model_name_or_path bigscience/T0_3B --output_dir "${OPEN_RESULTS_DIR}bigbench-$1" --parallelize --pad_to_max_length
    wait
}

# Open Response
BigBenchOpen auto_debugging
BigBenchOpen conlang_translation
BigBenchOpen linguistics_puzzles
BigBenchOpen logic_grid_puzzle
BigBenchOpen operators
BigBenchOpen parsinlu_reading_comprehension
BigBenchOpen repeat_copy_logic

# Deactivates conda environment
conda deactivate