import os
import copy
import pandas as pd
import numpy as np
from csv import DictWriter


def evaluate_detailed(queries, models, t, reranker_dir, results_path):
    # dictionary to store RaB/ARaB/nDRaB per model and query
    query_dict = {key: {'P': None, 'CP': None, 'P-CP': None} for key in queries.keys()}
    res = {'RaB': {key: copy.deepcopy(query_dict) for key in models},
           'ARaB': {key: copy.deepcopy(query_dict) for key in models},
           'nDRaB': {key: copy.deepcopy(query_dict) for key in models}}
    # dictionary to store RaB/ARaB/nDRaB per model and topic
    DocIndex = sorted(set([key[0] for key in queries.keys()]))  # set of all topics
    topic_dict = {key: {'P': 0, 'CP': 0, 'P-CP': 0} for key in DocIndex}  # dict per topic
    res_topic = {'RaB': {key: copy.deepcopy(topic_dict) for key in models},
                 'ARaB': {key: copy.deepcopy(topic_dict) for key in models},
                 'nDRaB': {key: copy.deepcopy(topic_dict) for key in models}}
    # dictionary to store RaB/ARaB/nDRaB per model
    res_averaged = {'RaB': {key: {'P': 0, 'CP': 0, 'P-CP': 0} for key in models},
                    'ARaB': {key: {'P': 0, 'CP': 0, 'P-CP': 0} for key in models},
                    'nDRaB': {key: {'P': 0, 'CP': 0, 'P-CP': 0} for key in models}}
    # get number of queries per topic to average over
    queries_per_topic = {key: 0 for key in DocIndex}
    for key in queries.keys():
        topic = key[0]
        queries_per_topic[topic] += 1

    # function to store values in dicts
    def values_to_dicts(method, model, key, topic, val_P, val_CP, val_PCP):
        res[method][model][key]['P'] = val_P
        res[method][model][key]['CP'] = val_CP
        res[method][model][key]['P-CP'] = val_PCP
        # per topic
        res_topic[method][model][topic]['P'] += val_P / queries_per_topic[topic]
        res_topic[method][model][topic]['CP'] += val_CP / queries_per_topic[topic]
        res_topic[method][model][topic]['P-CP'] += val_PCP / queries_per_topic[topic]
        # over all
        res_averaged[method][model]['P'] += val_P / (queries_per_topic[topic] * len(DocIndex))
        res_averaged[method][model]['CP'] += val_CP / (queries_per_topic[topic] * len(DocIndex))
        res_averaged[method][model]['P-CP'] += val_PCP / (queries_per_topic[topic] * len(DocIndex))

    # nDRaB weighting_factor
    weight = 1 / np.log2(np.arange(1, t + 1) + 1)  # max with this weight: 0.4543559338088345

    for key, query in queries.items():
        # load reranker results
        path = os.path.join(reranker_dir, 'results_' + key + '.tsv')
        df = pd.read_csv(path, sep='\t')
        topic = key[0]

        for model in models:
            df_model = df[df['tag'] == model]
            # get top 10 docno of each query variation
            df_N = df_model[df_model['qid'] == key + '_N']
            top_N = df_N[0:t]['docno'].tolist()
            df_P = df_model[df_model['qid'] == key + '_P']
            top_P = df_P[0:t]['docno'].tolist()
            df_CP = df_model[df_model['qid'] == key + '_CP']
            top_CP = df_CP[0:t]['docno'].tolist()

            # check if doc in top_N is also in top_P/top_CP
            N_P = np.array([1 if element in top_P else 0 for element in top_N])
            N_CP = np.array([1 if element in top_CP else 0 for element in top_N])

            # Rank Bias
            RaB_P = np.mean(N_P)
            RaB_CP = np.mean(N_CP)
            RaB = RaB_P - RaB_CP
            # store
            values_to_dicts('RaB', model, key, topic, RaB_P, RaB_CP, RaB)

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
            values_to_dicts('ARaB', model, key, topic, ARaB_P, ARaB_CP, ARaB)

            # normalized Discounted Rank Bias
            DRaB_P = np.mean(N_P * weight)
            DRaB_CP = np.mean(N_CP * weight)
            nDRaB_P = np.mean(N_P * weight) / np.mean(weight)
            nDRaB_CP = np.mean(N_CP * weight) / np.mean(weight)
            nDRaB = (DRaB_P - DRaB_CP) / np.mean(weight)
            # store
            values_to_dicts('nDRaB', model, key, topic, nDRaB_P, nDRaB_CP, nDRaB)

    # save
    for i, df in enumerate([res, res_topic]):
        if i == 0:
            element = 'query'
            path = os.path.join(results_path, 'res_per_query_t' + str(t) + '.csv')
        elif i == 1:
            element = 'topic'
            path = os.path.join(results_path, 'res_per_topic_t' + str(t) + '.csv')
        fields = ['metric', 'model', element, 'P', 'CP', 'P-CP']
        with open(path, 'w+') as f:
            writer = DictWriter(f, fields)
            writer.writeheader()
            for metric, nested_dict in df.items():
                for model, nested_dict in nested_dict.items():
                    for query, nested_dict in nested_dict.items():
                        row = {'metric': metric, 'model': model, element: query}
                        row.update(nested_dict)
                        writer.writerow(row)
    # save averaged
    path = os.path.join(results_path, 'res_averaged_t' + str(t) + '.csv')
    fields = ['metric', 'model', 'P', 'CP', 'P-CP']
    with open(path, 'w+') as f:
        writer = DictWriter(f, fields)
        writer.writeheader()
        for metric, nested_dict in res_averaged.items():
            for model, nested_dict in nested_dict.items():
                row = {'metric': metric, 'model': model}
                row.update(nested_dict)
                writer.writerow(row)


def merge_results(t_list, results_path):
    dfs_query = []
    dfs_topic = []
    dfs_averaged = []
    for t in t_list:
        df_query = pd.read_csv(os.path.join(results_path, 'res_per_query_t' + str(t) + '.csv'), sep=',')
        df_topic = pd.read_csv(os.path.join(results_path, 'res_per_topic_t' + str(t) + '.csv'), sep=',')
        df_averaged = pd.read_csv(os.path.join(results_path, 'res_averaged_t' + str(t) + '.csv'), sep=',')

        df_query['rank'] = t
        df_topic['rank'] = t
        df_averaged['rank'] = t

        dfs_query.append(df_query)
        dfs_topic.append(df_topic)
        dfs_averaged.append(df_averaged)

    merged_df_query = pd.concat(dfs_query)
    merged_df_topic = pd.concat(dfs_topic)
    merged_df_averaged = pd.concat(dfs_averaged)

    merged_df_query.to_csv(os.path.join(results_path, 'res_per_query' + '.csv'), index=False, sep=',')
    merged_df_topic.to_csv(os.path.join(results_path, 'res_per_topic' + '.csv'), index=False, sep=',')
    merged_df_averaged.to_csv(os.path.join(results_path, 'res_averaged' + '.csv'), index=False, sep=',')
