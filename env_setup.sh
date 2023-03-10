# Sets up pip3 and adds conda to PATH
export PATH="$PATH:/koko/system/anaconda/bin"
REPLICATION_DIR=~/Capstone/generate_all
EVAL_SCRIPT_SCORING=~/Capstone/generate_all/t0_multiQ.py
EVAL_SCRIPT_OPEN=~/Capstone/generate_all/t0_open.py

# Sets up directories for results
RESULTS_LOCATION=/common/users/ar1570
mkdir ${RESULTS_LOCATION}/capstone_rst
mkdir ${RESULTS_LOCATION}/capstone_rst/mc
mkdir ${RESULTS_LOCATION}/capstone_rst/open
SCORING_RESULTS_DIR=${RESULTS_LOCATION}/capstone_rst/mc/
OPEN_RESULTS_DIR=${RESULTS_LOCATION}/capstone_rst/open/

# Sets up path to conda env and to BIG-Bench
CONDA_DIR=/common/users/ar1570
ENV_NAME=mypython37
PATH_TO_T0=~/Capstone/t0_replication/rte/
BIG_BENCH=/common/users/ar1570/BIG-bench

# Sets up the conda environment itself
conda create -p ${CONDA_DIR}/${ENV_NAME} python=3.7
source activate ${CONDA_DIR}/${ENV_NAME}
conda install pip
${CONDA_DIR}/${ENV_NAME}/bin/pip3 install datasets ${PATH_TO_T0}t-zero --no-binary=protobuf protobuf==3.20.* ${BIG_BENCH}