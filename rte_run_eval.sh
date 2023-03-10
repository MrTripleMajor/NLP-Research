# Sets up pip3 and adds conda to PATH
export PATH="$PATH:/koko/system/anaconda/bin"
REPLICATION_DIR=~/Capstone/t0_replication/rte
EVAL_SCRIPT=${REPLICATION_DIR}/t-zero/evaluation/run_eval.py
RESULTS_DIR=${REPLICATION_DIR}/results_
CONDA_DIR=/data/local
ENV_NAME=mypython37

# Sets up conda environment
conda create -p ${CONDA_DIR}/${ENV_NAME} python=3.7
source activate ${CONDA_DIR}/${ENV_NAME}
conda install pip
${CONDA_DIR}/${ENV_NAME}/bin/pip install ${REPLICATION_DIR}/t-zero protobuf==3.20.*

# Iterates across templates
for template in "MNLI crowdsource", "guaranteed true", "can we infer", "GPT-3 style", "does this imply", "should assume", "does it follow that", "based on the previous passage", "justified in saying", "must be true"
do
    python ${EVAL_SCRIPT} --dataset_name super_glue --dataset_config_name rte --template_name "${template}" --model_name_or_path bigscience/T0_3B --output_dir "${RESULTS_DIR}${template}" --parallelize
    wait
done

# Deactivates conda environment
conda deactivate
