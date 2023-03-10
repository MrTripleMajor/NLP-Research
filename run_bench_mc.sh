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
BigBenchScoring ( ) {
    python ${EVAL_SCRIPT_SCORING} --dataset_name bigbench --dataset_config_name $1 --model_name_or_path bigscience/T0_3B --output_dir "${SCORING_RESULTS_DIR}bigbench-$1" --parallelize
    wait
}
BigBenchOpen ( ) {
    python ${EVAL_SCRIPT_OPEN} --dataset_name bigbench --dataset_config_name $1 --model_name_or_path bigscience/T0_3B --output_dir "${OPEN_RESULTS_DIR}bigbench-$1" --parallelize
    wait
}

# Scoring
#BigBenchScoring bbq_lite_json
#BigBenchScoring code_line_description
#BigBenchScoring conceptual_combinations
#BigBenchScoring emoji_movie
#BigBenchScoring formal_fallacies_syllogisms_negation
#BigBenchScoring hindu_knowledge
#BigBenchScoring known_unknowns
#BigBenchScoring language_identification
#BigBenchScoring logical_deduction
BigBenchScoring misconceptions_russian --per_device_eval_batch_size 2       # 16 examples, run with smaller batch size
#BigBenchScoring novel_concepts
#BigBenchScoring play_dialog_same_or_different
#BigBenchScoring strange_stories
#BigBenchScoring strategyqa                                                 # Needs its own parser
#BigBenchScoring symbol_interpretation
#BigBenchScoring vitaminc_fact_verification
#BigBenchScoring winowhy

# Deactivates conda environment
conda deactivate