import json
import pandas as pd
from os import listdir

names = []
acc = []

csv_names = []
inputs = []
predictions = []
targets = []

for subdir in listdir('generated-results/BIG-Bench/mc'):
    try:
        j = json.load(open('generated-results/BIG-Bench/mc/' + subdir + '/results.json'))
    except:
        continue
    print(len(j['inputs']), len(j['predictions']), len(j['targets']))
    names.append(j['dataset_name'] + '-' + j['dataset_config_name'])
    csv_names.extend([j['dataset_name'] + '-' + j['dataset_config_name']] * len(j['inputs']))
    inputs.extend(j['inputs'])
    predictions.extend(j['predictions'])
    targets.extend(j['targets'])
    acc.append(j['evaluation']['accuracy'])

pd.DataFrame({'dataset' : csv_names, 'inputs' : inputs, 'predictions' : predictions, 'targets' : targets}).to_csv('fullResults.csv')
pd.DataFrame({'dataset' : names, 'accuracy' : acc}).to_csv('summary.csv')