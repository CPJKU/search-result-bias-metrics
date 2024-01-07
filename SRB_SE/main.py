''' Script for loading search results from search engines and calculating the metrics (Rab, ARaB, nDRaB) according to ComSRB '''

import os
import copy
from csv import DictWriter

import pandas as pd
import numpy as np

# File containing search results
path = os.path.join('data', 'se_results.csv')
df = pd.read_csv(path, sep=';')

models = ['Google', 'Bing', 'DuckDuckGo']
queries = ['0_F', '1_F', '2_F', '3_F', '4_M', '5_M', '6_M', '7_M']
t = 10
weight = 1 / np.log2(np.arange(1, t + 1) + 1)

query_dict = {key: {'P': 0, 'CP': 0, 'P-CP': 0} for key in queries + ['all']}  # dict per topic
res = {'RaB': {key: copy.deepcopy(query_dict) for key in models},
       'ARaB': {key: copy.deepcopy(query_dict) for key in models},
       'nDRaB': {key: copy.deepcopy(query_dict) for key in models}}


def values_to_dicts(method, model, key, val_P, val_CP, val_PCP):
    res[method][model][key]['P'] = val_P
    res[method][model][key]['CP'] = val_CP
    res[method][model][key]['P-CP'] = val_PCP
    # over all
    res[method][model]['all']['P'] += val_P / len(queries)
    res[method][model]['all']['CP'] += val_CP / len(queries)
    res[method][model]['all']['P-CP'] += val_PCP / len(queries)


for model in models:
    df_model = df[df['tag'] == model]

    for key in queries:

        df_N = df_model[df_model['qid'] == key + '_N']
        top_N = df_N['docurl'].tolist()
        df_P = df_model[df_model['qid'] == key + '_P']
        top_P = df_P['docurl'].tolist()
        df_CP = df_model[df_model['qid'] == key + '_CP']
        top_CP = df_CP['docurl'].tolist()

        N_P = np.array([1 if element in top_P else 0 for element in top_N])
        N_CP = np.array([1 if element in top_CP else 0 for element in top_N])

        # Rank Bias
        RaB_P = np.mean(N_P)
        RaB_CP = np.mean(N_CP)
        RaB = RaB_P - RaB_CP
        # store
        values_to_dicts('RaB', model, key, RaB_P, RaB_CP, RaB)

        # Average Rank Bias
        RaB_P_x = np.zeros_like(N_P, dtype=float)
        RaB_CP_x = np.zeros_like(N_CP, dtype=float)
        for x in range(0, t):
            RaB_P_x[x] = np.mean(N_P[0:x + 1])
            RaB_CP_x[x] = np.mean(N_CP[0:x + 1])

        ARaB_P = np.mean(RaB_P_x)
        ARaB_CP = np.mean(RaB_CP_x)
        ARaB = ARaB_P - ARaB_CP
        # store
        values_to_dicts('ARaB', model, key, ARaB_P, ARaB_CP, ARaB)

        # normalized Discounted Rank Bias
        DRaB_P = np.mean(N_P * weight)
        DRaB_CP = np.mean(N_CP * weight)
        nDRaB_P = np.mean(N_P * weight) / np.mean(weight)
        nDRaB_CP = np.mean(N_CP * weight) / np.mean(weight)
        nDRaB = (DRaB_P - DRaB_CP) / np.mean(weight)
        # store
        values_to_dicts('nDRaB', model, key, nDRaB_P, nDRaB_CP, nDRaB)

# Save results
path = os.path.join(os.getcwd(), 'results.csv')
fields = ['metric', 'model', 'query', 'P', 'CP', 'P-CP']
with open(path, 'w+') as f:
    writer = DictWriter(f, fields)
    writer.writeheader()
    for method, nested_dict in res.items():
        for model, nested_dict in nested_dict.items():
            for query, nested_dict in nested_dict.items():
                row = {'metric': method, 'model': model, 'query': query}
                row.update(nested_dict)
                writer.writerow(row)
