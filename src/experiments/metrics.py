import numpy as np
from sklearn.preprocessing import MinMaxScaler

from preprocessing.constants import NUM_RESULTS, RESULT_COLUMNS


def get_average_ndcg(results):
    discount = 1 / np.log2(np.arange(NUM_RESULTS) + 2)
    dcgs = results.to_numpy() * discount
    idcgs = -np.sort(-results.to_numpy(), axis=1) * discount  # negatives for descending order
    ndcgs = dcgs.sum(axis=1) / idcgs.sum(axis=1)
    return np.nan_to_num(ndcgs).mean()


def normalize_relevance_values(df, min_val, max_val):
    mms = MinMaxScaler(feature_range=(min_val, max_val))
    return mms.fit_transform(df)


def sort_results(df):
    df.values.sort()
    df = df.iloc[:, ::-1]  # hack for sorting descending order since .sort has no descending arg
    df.columns = RESULT_COLUMNS
    return df
