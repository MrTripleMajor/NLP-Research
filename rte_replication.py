import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from parallelformers import parallelize

def prompt_template_1(premise, hypothesis):
    return premise + ' Using only the above description and what you know about the world, is ' + hypothesis + ' definitely correct? Yes or no?'

def prompt_template_2(premise, hypothesis):
    return 'Given ' + premise + ' Is it guaranteed true that ' + hypothesis + '? Yes or no?'

def prompt_template_3(premise, hypothesis):
    return 'Suppose ' + premise + ' Can we infer that ' + hypothesis + '? Yes or no?'

def prompt_template_4(premise, hypothesis):
    return premise + '\nQuestion: ' + hypothesis + ' True or False?'

def prompt_template_5(premise, hypothesis):
    return premise + '\n\nQuestion: Does this imply that \"' + hypothesis + '\"? Yes or no?' 

def prompt_template_6(premise, hypothesis):
    return 'Given ' + premise + ' Should we assume that \"' + hypothesis + '\" is true? Yes or no?'

def prompt_template_7(premise, hypothesis):
    return 'Given that ' + premise + ' Does it follow that ' + hypothesis + ' Yes or no?'

def prompt_template_8(premise, hypothesis):
    return premise + ' Based on the previous passage, is it true that \"' + hypothesis + '\"? Yes or no?'

def prompt_template_9(premise, hypothesis):
    return premise + ' Are we justified in saying that \"' + hypothesis + '\"? Yes or no?'

def prompt_template_10(premise, hypothesis):
    return 'Given that ' + premise + ' Therefore, it must be true that \"' + hypothesis + '\"? Yes or no?'

prompt_funcs = [prompt_template_1, prompt_template_2, prompt_template_3, prompt_template_4, prompt_template_5, prompt_template_6, prompt_template_7, prompt_template_8, \
                    prompt_template_9, prompt_template_10]



if __name__=='__main__':

    # Gets sets of prompts
    prompt_sets = [[], [], [], [], [], [], [], [], [], []]
    rte = load_dataset('super_glue', 'rte')
    for i in range(10):
        for example in rte['validation']:
            prompt_sets[i].append(prompt_funcs[i](example['premise'], example['hypothesis']))
    tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp")
    model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp")
    parallelize(model, num_gpus=8, fp16=True, verbose='detail')
    df = pd.DataFrame(prompt_sets)
    print(df.shape)
    df.to_csv('rte_prompts.csv')

    # Gets responses from prompts, and writes to CSV
    responses = [[], [], [], [], [], [], [], [], [], []]
    for i in range(10):
        for j in range(len(prompt_sets[i])):
            inputs = tokenizer(prompt_sets[i][j], return_tensors="pt")
            output = model.generate(**inputs, num_beams=5, no_repeat_ngram_size=4, max_length=15)
            responses[i].append(tokenizer.batch_decode(output, skip_special_tokens = True))
    df = pd.DataFrame(responses)
    print(df.shape)
    df.to_csv('rte_responses.csv')

    # Parses responses
    for i in range(10):
        for j in range(len(responses[i])):
            if 'Yes' in responses[i][j] or 'True' in responses[i][j]:
                responses[i][j] = 0
            elif 'No' in responses[i][j] or 'False' in responses[i][j]:
                responses[i][j] = 1
            else:
                responses[i][j] = -1
    accuracy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(10):
        for j in range(len(responses[i])):
            if (responses[i][j] == rte['validation'][j]['label']):
                accuracy[i] += 1
        accuracy[i] /= len(responses[i])
    df = pd.DataFrame(accuracy)
    print(df.shape)
    df.to_csv('rte_response_accuracies.csv')
