# coding=utf-8

"""
Script showcasing how to run inference of T0++ on multiple GPUs using model parallelism. The model will be splitted across all available devices.
Note that this feature is still an experimental feature under 🤗 Transformers.
The minimum requirements to run T0++ (11B parameters) inference are 4 16GB V100 or 2 32GB V100 (or basically, enough GPU memory to fit ~42GB of fp32 parameters).
"""

import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoConfig,
    AutoTokenizer,
)

import datasets
from datasets import load_dataset, load_metric

import torch

import argparse
from accelerate import Accelerator
import logging
import json
import os

logger = logging.getLogger(__name__)



def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce main evaluation in T0.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
        required=True,
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--template_name",
        type=str,
        default=None,
        help="The template/prompt name",
        required=False,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--target_max_length",
        type=int,
        default=256,
        help="Target max length. Sequences longer than this will be truncated."
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models. The list of T0 variants can be found on `https://huggingface.co/bigscience/T0_3B`",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--parallelize",
        action="store_true",
        help=(
            "If passed, will call `model.parallelize` which splits the model on all GPUs available when applicable (model parallelism). "
            "Note that this feature is still experimental in HF Transformers."
        ),
    )
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()


    # Handle the output directory creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # In distributed evaluation, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if args.dataset_name == "anli":
            error_message = "For ANLI, `dataset_config_name` should be either `dev_r1`, `dev_r2` or `dev_r3`."
            assert args.dataset_config_name is not None, error_message
            assert args.dataset_config_name in ["dev_r1", "dev_r2", "dev_r3"], error_message
            raw_datasets = load_dataset(args.dataset_name, split=args.dataset_config_name)
        else:
            raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, split="validation")
    #TODO(Victor): enable loading pre-processed dataset from https://huggingface.co/datasets/bigscience/P3

    # Trim a number of evaluation examples
    if args.debug:
        raw_datasets = raw_datasets.select(range(min(len(raw_datasets),100)))

    column_names = raw_datasets.column_names


    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "Either `args.config_name` or `args.model_name_or_path` should be provided."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer, padding_side="left")
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, padding_side="left")
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if tokenizer.pad_token is None:
        for token in [tokenizer.eos_token, tokenizer.bos_token, tokenizer.sep_token]:
            if token is not None:
                tokenizer.pad_token = token
        if tokenizer.pad_token is None:
            raise ValueError("Please define a pad token id.")

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)
    if args.parallelize:
        model.parallelize()

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False
    
    results = {
        "inputs"        : [],
        "predictions"   : [],
        "targets"       : [],
    }

    model.eval()
    results['inputs'] = raw_datasets['inputs']
    results['targets'] = raw_datasets['targets']
    for i in range(len(raw_datasets['inputs'])):
        inputs = tokenizer.encode(raw_datasets['inputs'][i], return_tensors="pt")
        inputs = inputs.to("cuda:0")
        with torch.no_grad():
            outputs = model.generate(inputs)
        results['predictions'].append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    results["dataset_name"] = args.dataset_name
    results["dataset_config_name"] = args.dataset_config_name
    results["template_name"] = args.template_name

    if accelerator.is_main_process:
        if args.output_dir is not None:
            with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=4)