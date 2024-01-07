import os
import pandas as pd

def trec_run_files(queries_N, models, dir, reranker_dir, dir_trec_files):
    # reranker results
    dfs = []
    for key in queries_N.keys():
        # load and append
        path = os.path.join(reranker_dir, 'results_' + key[:-2] + '.tsv')
        df = pd.read_csv(path, sep='\t')
        df = df[df['qid'].str.endswith('N')]
        dfs.append(df)
    df = pd.concat(dfs)
    #clean
    df.drop(columns=['passage'], inplace=True)
    df['qid'] = df['qid']
    #split according to model
    for i, model in enumerate(models):
        df_m = df[df['tag']==model]
        #save
        save_path = os.path.join(dir_trec_files, 'results_' + model + '.run')
        df_m.to_csv(save_path, index=False, header=False, sep=' ')

def queries_gender_annotated(queries_N, dir):
    # queries_gender_annotated file
    df_anno = pd.DataFrame.from_dict(queries_N, orient='index')
    df_anno['annotation'] = 'n'
    df_anno.to_csv(os.path.join(dir, 'resources', 'queries_gender_annotated.csv'), header=None)
    # individual files for each query
    DocIndex = sorted(set([key[0] for key in queries_N.keys()]))
    for i in DocIndex:
        df_anno_topic = df_anno[df_anno.index.str[0] == i]
        df_anno_topic.to_csv(os.path.join(dir, 'resources', 'queries_gender_annotated_'+ i + '.csv'), header=None)