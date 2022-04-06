import os
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing.preprocessing import convert_raw_data_to_product_records
from preprocessing.constants import RESULT_COLUMNS, NUM_RESULTS, DATA_DIRECTORY, TRAIN_FILE
from metrics import get_average_ndcg

RESULT_IDS = pd.read_csv(os.path.join('..', DATA_DIRECTORY, 'results_by_query.csv'), index_col='search_term')
TRAIN_DATA = pd.read_csv(os.path.join('..', DATA_DIRECTORY, TRAIN_FILE), encoding='ISO-8859-1')

TOKENIZER = BertTokenizer.from_pretrained('prajjwal1/bert-small')
BERT = BertModel.from_pretrained('prajjwal1/bert-small')


def precompute_embeddings(product_dict):
    print('Precomputing embeddings for dataset')
    for i, (uid, product) in enumerate(product_dict.items()):
        title_and_desc = product['product_title'] + ' ' + product['product_description']
        all_text = title_and_desc + ' ' + product.get('attributes', '')
        title_desc_seq = TOKENIZER.encode(title_and_desc, truncation=True, max_length=512, return_tensors='pt')
        all_text_seq = TOKENIZER.encode(all_text, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            title_desc_emb = BERT(title_desc_seq, output_hidden_states=True).hidden_states[0]
            all_text_emb = BERT(all_text_seq, output_hidden_states=True).hidden_states[0]
        product['title_desc_emb'] = title_desc_emb.mean(axis=1).numpy()
        product['all_text_emb'] = all_text_emb.mean(axis=1).numpy()
        if i % 1000 == 0:
            print(f'Finished embedding for {i} documents')
    return product_dict


def get_sim(val, emb, query_emb, product_dict):
    if val == 0 or val not in product_dict:
        return 0
    return cosine_similarity(product_dict[val][emb], query_emb)[0][0]


def convert_to_similarities(row, emb, product_dict):
    query = row.name
    query_seq = TOKENIZER.encode(query, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        query_emb = BERT(query_seq, output_hidden_states=True).hidden_states[0]
        query_emb = query_emb.mean(axis=1)
    sims = row[RESULT_COLUMNS].apply(lambda val: get_sim(val, emb, query_emb, product_dict))
    return sims


def sort_results(df):
    df.values.sort()
    df = df.iloc[:, ::-1]  # hack for sorting descending order since .sort has no descending arg
    df.columns = RESULT_COLUMNS
    return df


def ids_to_relevance(row):
    search_term = row.name
    relevance_scores = [search_term]
    for col in RESULT_COLUMNS:
        uid = row[col]
        relevance_row = TRAIN_DATA[(TRAIN_DATA['search_term'] == search_term) & (TRAIN_DATA['product_uid'] == uid)]
        if relevance_row.empty:
            relevance_scores.append(0.0)
        else:
            relevance_scores.append(relevance_row['relevance'].iloc[0])
    return pd.Series(relevance_scores)


def sort_by_sim(row, sims_df):
    sims = sims_df.loc[row.name]
    return row[np.argsort(sims)].values


if __name__ == '__main__':
    product_records = convert_raw_data_to_product_records()
    product_dict = {}
    for rec in product_records:
        uid = rec['product_uid']
        product_dict[uid] = rec
    product_dict = precompute_embeddings(product_dict)

    # compute relevance to the query based on embedding similarity
    title_sims = RESULT_IDS.apply(lambda row: convert_to_similarities(row, 'title_desc_emb', product_dict), axis=1)
    full_text_sims = RESULT_IDS.apply(lambda row: convert_to_similarities(row, 'all_text_emb', product_dict), axis=1)

    # sort the result ids based on the computed similarities
    title_sims_ids = pd.DataFrame(RESULT_IDS.apply(lambda row: sort_by_sim(row, title_sims), axis=1).to_list(), index=RESULT_IDS.index, columns=RESULT_COLUMNS)
    full_text_sims_ids = pd.DataFrame(RESULT_IDS.apply(lambda row: sort_by_sim(row, full_text_sims), axis=1).to_list(), index=RESULT_IDS.index, columns=RESULT_COLUMNS)

    title_sims_ids.to_csv(os.path.join('..', DATA_DIRECTORY, 'title_bert_rankings.csv'), header=True, index=True)
    full_text_sims_ids.to_csv(os.path.join('..', DATA_DIRECTORY, 'full_text_bert_rankings.csv'), header=True, index=True)

    # get the original relevance scores from the train dataset
    title_relevance = title_sims_ids.apply(ids_to_relevance, axis=1)
    full_text_relevance = full_text_sims_ids.apply(ids_to_relevance, axis=1)
    # pandas housekeeping
    title_relevance.columns = ['search_term'] + RESULT_COLUMNS
    full_text_relevance.columns = ['search_term'] + RESULT_COLUMNS
    title_relevance.set_index('search_term', inplace=True)
    full_text_relevance.set_index('search_term', inplace=True)

    # calculate ndcg
    title_ndcg = get_average_ndcg(title_relevance)
    full_text_ndcg = get_average_ndcg(full_text_relevance)
    print(f'Average NDCG using title and description: {title_ndcg}')
    print(f'Average NDCG using title, description, and attributes: {full_text_ndcg}')
