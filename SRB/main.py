''' Main script for loading models and running ComSRB tests (+ preparing data for RepSRB tests) '''

import os
import pandas as pd
import json

from BM25 import BM25_retrieval
from re_rankers import re_rank # re-rankers and function for re-ranking
from evaluation import evaluate # function for evaluating ranked collections
from evaluation_detailed import evaluate_detailed, merge_results # functions for detailed evaluation
from prepare_RepSRB import queries_gender_annotated, trec_run_files # functions to prepare data for RepSRB


queries_file = 'retrieval_queries' # queries in three variations in the order N,P,CP !!!
n_hits = 1000 # number of docs to retrieve using BM25 per query variation

data_dir = 'data'
BM25_dir = 'BM25_results'
collection_path = os.path.join('MSMARCO', 'collection.tsv')

reranker_dir = 'reranker_results'
results_dir = 'results'
results_dir_detailed = os.path.join(results_dir, 'detailed')

# For data preparation for RepSRB
dir_rep = 'RepSRB'
dir_trec_files = os.path.join(dir_rep, 'reranker_trec_run')

for directory_path in [BM25_dir, reranker_dir, results_dir, dir_trec_files, results_dir_detailed]:
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        os.makedirs(directory_path)


# MODELS
# model_name->[type,path]
models = {'bi_msmarco_L6': ['bi', os.path.join('sentence-transformers/msmarco-MiniLM-L6-cos-v5')],
          'bi_msmarco_L12': ['bi', os.path.join('sentence-transformers/msmarco-MiniLM-L12-cos-v5')],
          'cross_msmarco_L6': ['cross',  os.path.join('cross-encoder/ms-marco-MiniLM-L-6-v2')],
          'cross_msmarco_L12': ['cross', os.path.join('cross-encoder/ms-marco-MiniLM-L-12-v2')]}

model_names = list(models.keys())


## ComSRB Experiments

# load the queries
path = os.path.join(data_dir, queries_file + '.jsonl')
queries_import = json.load(open(path, 'r'))
queries = pd.json_normalize(queries_import, sep='_').to_dict(orient='records')[0]

# first-stage ranker
BM25_retrieval(queries, n_hits, collection_path, BM25_dir)

# re-ranker
re_rank(queries, BM25_dir, reranker_dir, models)

# evaluation (only one cutoff (t), and results are averaged per topic)
results_path_t10 = os.path.join(results_dir,'res.csv')
evaluate(queries, model_names, 10, reranker_dir, results_path_t10)

# detailed evaluation (results averaged, per topic and per query; multiple cutoffs (t) can be specified)
evaluate_detailed(queries, model_names, 10, reranker_dir, results_dir_detailed)
evaluate_detailed(queries, model_names, 5, reranker_dir, results_dir_detailed)
evaluate_detailed(queries, model_names, 20, reranker_dir, results_dir_detailed)

merge_results([5, 10, 20], results_dir_detailed)


## Prepare data for RepSRB Experiments (to match the required input of the Jupyter Notebook)

# only neutral queries
queries_N = {str(key) + '_N' : val[0:1] for key, val in queries.items()}

# gender annotated queries (also per topic)
queries_gender_annotated(queries_N, dir_rep)
# create trec_run_files with re-ranked results
trec_run_files(queries_N, model_names, dir_rep, reranker_dir, dir_trec_files)







