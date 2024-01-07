import os
import copy
import itertools

import pandas as pd
import numpy as np
import scipy.stats as stats

from csv import DictWriter


def evaluate(queries, models, t, reranker_dir, results_path):
    # dictionary to store RaB/ARaB/nDRaB per model and query
    DocIndex = sorted(set([key[0] for key in queries.keys()]))  # set of all topics
    topic_dict = {key: [] for key in DocIndex}  # dict per topic
    res = {'RaB': {key: copy.deepcopy(topic_dict) for key in models},
           'ARaB': {key: copy.deepcopy(topic_dict) for key in models},
           'nDRaB': {key: copy.deepcopy(topic_dict) for key in models}}

    # nDRaB weighting_factor
    weight = 1 / np.log2(np.arange(1, t + 1) + 1)

    for key, query in queries.items():
        # load reranker results
        path = os.path.join(reranker_dir, 'results_' + key + '.tsv')
        df = pd.read_csv(path, sep='\t')
        topic = key[0]

        for model in models:
            df_model = df[df['tag'] == model]
            # get top t docno of each query variation
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

            res['RaB'][model][topic].append(RaB)

            # Average Rank Bias
            RaB_P_x = np.zeros_like(N_P, dtype=float)
            RaB_CP_x = np.zeros_like(N_CP, dtype=float)
            for x in range(0, t):
                RaB_P_x[x] = np.mean(N_P[0:x + 1])
                RaB_CP_x[x] = np.mean(N_CP[0:x + 1])

            ARaB_P = np.mean(RaB_P_x)
            ARaB_CP = np.mean(RaB_CP_x)
            ARaB = ARaB_P - ARaB_CP

            res['ARaB'][model][topic].append(ARaB)

            # normalized Discounted Rank Bias
            DRaB_P = np.mean(N_P * weight)
            DRaB_CP = np.mean(N_CP * weight)
            nDRaB = (DRaB_P - DRaB_CP) / np.mean(weight)

            res['nDRaB'][model][topic].append(nDRaB)

    # function to calculate test statistics
    def test_statistics(data, metric, topic):
        mean = np.mean(data)
        ttest = stats.ttest_1samp(a=data, popmean=0)
        pval = ttest.pvalue
        esize = (np.mean(data) - 0) / np.std(data, ddof=1)  # cohens
        print('Model: {}, Topic: {}'.format(model, topic))
        print('\tp-val: {:.5f}\tesize: {:.3f} \t{}: {:.3f}'.format(pval, esize, metric, mean))
        results.append(dict(
            topic=topic,
            metric=metric,
            model=model,
            mean=mean,
            effect_size=esize,
            p_value=pval,
            number_of_queries=len(data),
            rank=t))

    # perform one sample t-test
    results = []
    for model in models:
        for metric, metric_dict in res.items():
            # data for all topics in one list
            data = list(itertools.chain(*metric_dict[model].values()))
            test_statistics(data, metric, 'all topics')
            # data for each query
            for topic in DocIndex:
                data = metric_dict[model][topic]
                test_statistics(data, metric, topic)

    # save
    with open(results_path, 'w+') as f:
        writer = DictWriter(f, fieldnames=results[0].keys(), delimiter='\t')
        writer.writeheader()
        for r in results:
            writer.writerow(r)
