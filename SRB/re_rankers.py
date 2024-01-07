''' Script containig functions to re-rank the search results using bi- and cross-encoders from the sentence-transformers library '''

import os
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer, util

def cross_encoder(query, docs, df, version, tag):
    # prepare data
    pairs = [(query, d) for d in docs]

    # calculate scores
    model = CrossEncoder(version, max_length=512)
    scores = model.predict(pairs, batch_size=128)

    # store in dataframe
    df['score'] = scores
    df.sort_values(by=['score'], ascending=False, inplace=True)
    df['rank'] = range(1, len(df) + 1)  # add rank to sorted list
    df['tag'] = tag

    return df[['qid', 'Q0', 'docno', 'rank', 'score', 'tag', 'passage']]

def bi_encoder(query, docs, df, version, tag):
    model = SentenceTransformer(version)
    # embeddings
    doc_embeddings = model.encode(docs, convert_to_tensor=True, show_progress_bar=True, batch_size = 128)
    query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=True, batch_size = 128)
    # calculate scores
    scores = util.dot_score(query_embedding, doc_embeddings)[0].cpu().tolist()

    # store in dataframe
    df['score'] = scores
    df.sort_values(by=['score'], ascending=False, inplace=True)  # add rank to sorted list
    df['rank'] = range(1, len(df)+1)
    df['tag'] = tag

    return df[['qid', 'Q0', 'docno', 'rank', 'score', 'tag', 'passage']]


def re_rank(queries, BM25_dir, reranker_dir, models: dict):
    for key, query in queries.items():
        # get top passages for each query (as retrieved with BM25)
        path = os.path.join(BM25_dir, 'collection_' + key + '.tsv')
        df = pd.read_csv(path, sep='\t', header=None, names=['docno','passage'])
        passages = df["passage"].tolist()
        df['Q0'] = 'Q0'

        # empty df to store all results for one query
        df_results = pd.DataFrame(columns=['qid', 'Q0', 'docno', 'rank', 'score', 'tag', 'passage'])
        # iterate through query variations
        for i, q in enumerate(query):
            # query variation
            var = 'N' if i == 0 else 'P' if i == 1 else 'CP'
            df['qid'] = key + '_' + var
            # models
            for model_name, [model_type, model_path] in models.items():
                # Re-ranking procedure depending on model
                if model_type == 'bi':
                    df_m = bi_encoder(q, passages, df.copy(), model_path, model_name)
                elif model_type == 'cross':
                    df_m = cross_encoder(q, passages, df.copy(), model_path, model_name)
                df_results = pd.concat([df_results, df_m])

        # save
        path = os.path.join(reranker_dir, 'results_' + key + '.tsv')
        df_results.to_csv(path, index=False, sep='\t')

    print('Saved re-ranked results to:', reranker_dir)
