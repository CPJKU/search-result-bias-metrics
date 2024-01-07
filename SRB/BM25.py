''' Script for performing first-stage retrieval with BM25 using Pyserini '''

import os
from pyserini.search.lucene import LuceneSearcher

def BM25_retrieval(queries, n_hits, collection_path, save_dir='BM25_results'):
    searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
    # retrieval for each query
    for key, query in queries.items():
        ids = set()
        for query_variation in query: # each variation for one query (N,P,CP)
            hits = searcher.search(query_variation, n_hits)
            # get ids
            for i in range(0, n_hits):
                ids.add(int(hits[i].docid))

        tsvfile = open(collection_path, 'r').readlines()
        with open(os.path.join(save_dir, 'collection_' + key + '.tsv'), 'w+', newline='') as newfile:
            for i in ids:
                newfile.writelines(tsvfile[i])

    print('Saved results from BM25 retrieval to:', save_dir)