# Sets up pip3 and adds conda to PATH
export PATH="$PATH:/koko/system/anaconda/bin"
REPLICATION_DIR=~/Capstone/generate_all
EVAL_SCRIPT=~/Capstone/generate_all/t0_generate.py

mkdir /data/local/capstone_rst
RESULTS_DIR=/data/local/capstone_rst/results-

CONDA_DIR=/data/local
ENV_NAME=mypython37
PATH_TO_T0=~/Capstone/t0_replication/rte/

# Sets up conda environment
source activate ${CONDA_DIR}/${ENV_NAME}
conda install pip
${CONDA_DIR}/${ENV_NAME}/bin/pip install ${PATH_TO_T0}t-zero protobuf==3.20.*

# A function to run T0
Run_T0 ( ) {
    python ${EVAL_SCRIPT} --dataset_name $1 --dataset_config_name $2 --template_name "$3" --model_name_or_path bigscience/T0_3B --output_dir "${RESULTS_DIR}$1-$2-$3" --parallelize
    wait
}
Run_T0_hellaswag () {
    python ${EVAL_SCRIPT} --dataset_name hellaswag --template_name "$1" --model_name_or_path bigscience/T0_3B --output_dir "${RESULTS_DIR}hellaswag-$1" --parallelize
    wait
}

# super_glue & rte
Run_T0 super_glue rte "MNLI crowdsource"
Run_T0 super_glue rte "guaranteed true"
Run_T0 super_glue rte "can we infer"
Run_T0 super_glue rte "GPT-3 style"
Run_T0 super_glue rte "does this imply"
Run_T0 super_glue rte "should assume"
Run_T0 super_glue rte "does it follow that"
Run_T0 super_glue rte "based on the previous passage"
Run_T0 super_glue rte "justified in saying"
Run_T0 super_glue rte "must be true"

# super_glue & cb
Run_T0 super_glue cb "can we infer"
Run_T0 super_glue cb "based on the previous passage"
Run_T0 super_glue cb "claim true/false/inconclusive"
Run_T0 super_glue cb "does it follow that"
Run_T0 super_glue cb "justified in saying"
Run_T0 super_glue cb "always/sometimes/never"
Run_T0 super_glue cb "GPT-3 style"
Run_T0 super_glue cb "consider always/sometimes/never"
Run_T0 super_glue cb "guaranteed true"
Run_T0 super_glue cb "must be true"
Run_T0 super_glue cb "guaranteed/possible/impossible"
Run_T0 super_glue cb "does this imply"
Run_T0 super_glue cb "MNLI crowdsource"
Run_T0 super_glue cb "should assume"
Run_T0 super_glue cb "take the following as truth"

# ANLI & dev_r1
Run_T0 anli dev_r1 "MNLI crowdsource"
Run_T0 anli dev_r1 "should assume"
Run_T0 anli dev_r1 "does it follow that"
Run_T0 anli dev_r1 "GPT-3 style"
Run_T0 anli dev_r1 "based on the previous passage"
Run_T0 anli dev_r1 "justified in saying"
Run_T0 anli dev_r1 "take the following as truth"
Run_T0 anli dev_r1 "must be true"
Run_T0 anli dev_r1 "can we infer"
Run_T0 anli dev_r1 "guaranteed/possible/impossible"
Run_T0 anli dev_r1 "always/sometimes/never"
Run_T0 anli dev_r1 "does this imply"
Run_T0 anli dev_r1 "consider always/sometimes/never"
Run_T0 anli dev_r1 "claim true/false/inconclusive"
Run_T0 anli dev_r1 "guaranteed true"

# ANLI & dev_r2
Run_T0 anli dev_r2 "MNLI crowdsource"
Run_T0 anli dev_r2 "should assume"
Run_T0 anli dev_r2 "does it follow that"
Run_T0 anli dev_r2 "GPT-3 style"
Run_T0 anli dev_r2 "based on the previous passage"
Run_T0 anli dev_r2 "justified in saying"
Run_T0 anli dev_r2 "take the following as truth"
Run_T0 anli dev_r2 "must be true"
Run_T0 anli dev_r2 "can we infer"
Run_T0 anli dev_r2 "guaranteed/possible/impossible"
Run_T0 anli dev_r2 "always/sometimes/never"
Run_T0 anli dev_r2 "does this imply"
Run_T0 anli dev_r2 "consider always/sometimes/never"
Run_T0 anli dev_r2 "claim true/false/inconclusive"
Run_T0 anli dev_r2 "guaranteed true"

# ANLI & dev_r3
Run_T0 anli dev_r3 "MNLI crowdsource"
Run_T0 anli dev_r3 "should assume"
Run_T0 anli dev_r3 "does it follow that"
Run_T0 anli dev_r3 "GPT-3 style"
Run_T0 anli dev_r3 "based on the previous passage"
Run_T0 anli dev_r3 "justified in saying"
Run_T0 anli dev_r3 "take the following as truth"
Run_T0 anli dev_r3 "must be true"
Run_T0 anli dev_r3 "can we infer"
Run_T0 anli dev_r3 "guaranteed/possible/impossible"
Run_T0 anli dev_r3 "always/sometimes/never"
Run_T0 anli dev_r3 "does this imply"
Run_T0 anli dev_r3 "consider always/sometimes/never"
Run_T0 anli dev_r3 "claim true/false/inconclusive"
Run_T0 anli dev_r3 "guaranteed true"

# super_glue & wsc.fixed
Run_T0 super_glue wsc.fixed "does the pronoun refer to"
Run_T0 super_glue wsc.fixed "by p they mean"
Run_T0 super_glue wsc.fixed "in other words"
Run_T0 super_glue wsc.fixed "I think they mean"
Run_T0 super_glue wsc.fixed "does p stand for"
Run_T0 super_glue wsc.fixed "GPT-3 Style"
Run_T0 super_glue wsc.fixed "replaced with"
Run_T0 super_glue wsc.fixed "p is/are r"
Run_T0 super_glue wsc.fixed "the pronoun refers to"
Run_T0 super_glue wsc.fixed "Who or what is/are"

# winogrande & winogrande_x1
Run_T0 winogrande winogrande_xl "does underscore refer to"
Run_T0 winogrande winogrande_xl "stand for"
Run_T0 winogrande winogrande_xl "underscore refer to"
Run_T0 winogrande winogrande_xl "fill in the blank"
Run_T0 winogrande winogrande_xl "Replace"

# story_cloze & 2016
Run_T0 story_cloze 2016 "Answer Given options"
Run_T0 story_cloze 2016 "Choose Story Ending"
Run_T0 story_cloze 2016 "Movie What Happens Next"
Run_T0 story_cloze 2016 "Story Continuation and Options"
Run_T0 story_cloze 2016 "Novel Correct Ending"

# super_glue & wic
Run_T0 super_glue wic "question-context-meaning-with-label"
Run_T0 super_glue wic "question-context-meaning"
Run_T0 super_glue wic "grammar_homework"
Run_T0 super_glue wic "affirmation_true_or_false"
Run_T0 super_glue wic "GPT-3-prompt"
Run_T0 super_glue wic "same_sense"
Run_T0 super_glue wic "question-context"
Run_T0 super_glue wic "GPT-3-prompt-with-label"
Run_T0 super_glue wic "polysemous"
Run_T0 super_glue wic "similar-sense"

# hellaswag
Run_T0_hellaswag "Predict ending with hint"
Run_T0_hellaswag "Randomized prompts template"
Run_T0_hellaswag "complete_first_then"
Run_T0_hellaswag "if_begins_how_continues"

# super_glue & copa
Run_T0 super_glue copa "exercise"
Run_T0 super_glue copa "…What could happen next, C1 or C2?"
Run_T0 super_glue copa "i_am_hesitating"
Run_T0 super_glue copa "plausible_alternatives"
Run_T0 super_glue copa "C1 or C2? premise, so/because…"
Run_T0 super_glue copa "…As a result, C1 or C2?"
Run_T0 super_glue copa "best_option"
Run_T0 super_glue copa "…which may be caused by"
Run_T0 super_glue copa "more likely"
Run_T0 super_glue copa "cause_effect"
Run_T0 super_glue copa "…why? C1 or C2"
Run_T0 super_glue copa "choose"

# Deactivates conda environment
conda deactivate
