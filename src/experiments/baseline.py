import os
import numpy as np
import pandas as pd

from elastic_search.search import search_all_fields
from preprocessing.constants import DATA_DIRECTORY, TRAIN_FILE

NUM_RESULTS = 20
RESULT_COLUMNS = [f'result_{i}' for i in range(NUM_RESULTS)]

train_data = pd.read_csv(os.path.join('..', DATA_DIRECTORY, TRAIN_FILE), encoding='ISO-8859-1')


def get_top_ten_result_ids(query):
    response = search_all_fields(query, num_docs=len(RESULT_COLUMNS))
    top_ids = [int(r['_id']) for r in response['hits']['hits']]
    return top_ids


def ids_to_relevance(row):
    search_term = row['search_term']
    relevance_scores = [search_term]
    for col in RESULT_COLUMNS:
        uid = row[col]
        relevance_row = train_data[(train_data['search_term'] == search_term) & (train_data['product_uid'] == uid)]
        if relevance_row.empty:
            relevance_scores.append(0.0)
        else:
            relevance_scores.append(relevance_row['relevance'].iloc[0])
    return pd.Series(relevance_scores)


def get_average_ndcg(results):
    discount = 1 / np.log2(np.arange(NUM_RESULTS) + 2)
    dcgs = results.to_numpy() * discount
    idcgs = -np.sort(-results.to_numpy(), axis=1) * discount  # negatives for descending order
    ndcgs = dcgs.sum(axis=1) / idcgs.sum(axis=1)
    return np.nan_to_num(ndcgs).mean()


def get_baseline_results():
    search_terms = train_data['search_term'].drop_duplicates()
    result_id_lists = pd.DataFrame(search_terms.apply(get_top_ten_result_ids).to_list(), columns=RESULT_COLUMNS)
    result_id_lists.fillna(0, inplace=True)  # some queries have less than the max amount of results
    result_id_lists = result_id_lists.astype(int)
    results_by_query = pd.concat((search_terms.reset_index(drop=True), result_id_lists), axis=1)
    results_by_query.to_csv(os.path.join('..', DATA_DIRECTORY, 'results_by_query.csv'), header=True, index=False)
    print('Evaluating baseline relevance...')
    relevance_by_query = results_by_query.apply(ids_to_relevance, axis=1)
    relevance_by_query.columns = ['search_term'] + RESULT_COLUMNS
    average_ndcg = get_average_ndcg(relevance_by_query[RESULT_COLUMNS])
    print(f'Average NDCG for baseline system: {average_ndcg}')


if __name__ == '__main__':
    get_baseline_results()
