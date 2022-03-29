from elasticsearch import Elasticsearch, BadRequestError

from preprocessing.preprocessing import convert_raw_data_to_product_records


def initialize_elastic(port=9200):
    es = Elasticsearch(f'http://localhost:{port}')
    return es


def ingest_records(client, records):
    for i, record in enumerate(records):
        product_uid = record.pop('product_uid')
        try:
            resp = client.index(index='hd_index', id=product_uid, document=record)
        except BadRequestError as bre:
            print(f'Error indexing record')
            print(record)
            print(bre)
            continue
        if i % 1000 == 0:
            print(f'Indexed {i} records')
            print(resp)
    return client


if __name__ == '__main__':
    es_client = initialize_elastic()
    records = convert_raw_data_to_product_records()
    es_client = ingest_records(es_client, records)
    print(es_client.cat.count(index='hd_index', format='json'))
