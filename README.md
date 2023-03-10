# Capstone Project (2022-2023)
Title: Understanding the Zero-Shot Learning Capabilities of Pretrained Language Models

Recently, the field of natural language processing (NLP) has been galvanized by the seemingly astounding capabilities of large-scale pretrained language models (PLMs) to solve unknown tasks (i.e., zero-shot learning) as long as they can be framed in natural language (e.g., provide the input "Translate the following sentence to German: [sentence]" to a PLM, expect a German translation of [sentence] as output). This approach, also known as "prompting", has generated a number of papers trying to use and better understand it, but most are superficial applications and analyses which do not offer a satisfying explanation for why prompting works and how. In this project, we will thoroughly chart the current landscape of prompting prominent PLMs (e.g., GPT-3, T0) by systematic experiments on standard zero-shot performance benchmarks (e.g., T0 datasets) and aim to develop a new understanding and methods of prompting.

## BIG-Bench Lite Evaluation of T0

In the "big-bench-lite-test" folder, we are evaluating the 3B-parameter T0 model on the BIG-Bench Lite datasets.
- Current Progress:
  - 15 out of 17 multiple choice datasets evaluated
  - 7 out of 7 free-response datasets evaluated 👍
- In Progress:
  - Now that I was able to generate my own script and really hone in on how the Sanh et al script works, completing the script and experiment for both of the following should take about an hour or two, plus another hour or two to analyze results.
  - StrategyQA dataset: Needs its own parser. Work in progress. Not the same format as the rest of BIG-Bench. Maybe contact BIG-Bench administrators about it?
  - Russian Misconceptions dataset: Our current testing scripts fail for datasets this small. A non-parallelized version is being made.

To evaluate BIG-Bench Lite, we adapted the Sanh et al script for the multiple-choice (scoring) problems, with minor changes for performance and executability (without changes in results). We then realized that due to the preprocessing Sanh et al employ, their script does not work for the free-response (open-ended) problems. We thus decided to write our own script, drawing inspiration from their model parallelization sample script and the multiple-choice script. Thus, we maintain the same tokenization methods (and in theory the same results) as Sanh et al's scripts.

## Sanh et al Experiment Replication

In the "sanh_replication" folder, we performed a replication of Sanh et al's experiment on the 3B-parameter T0 model, obtaining the same results as in the paper. This experiment was built upon the scripts available on their [GitHub repo](https://github.com/bigscience-workshop/t-zero).

## Parallelformers Experiment

In the "parallelformers_rte" folder, we performed our own evaluation of RTE using the T0 model. Although we initially had differing results from Sanh et al, we later learned that this was due to the fact that Sanh et al used preprocessing which we did not use.
