from .ingest import initialize_elastic


es = initialize_elastic()


def search_all_fields(query_value, num_docs=10):
    # do some minor query cleaning
    query_value = query_value.replace('/', '\/') \
        .replace(':', '') \
        .replace('~', '\~')
    query = {
        'query_string': {
            'query': query_value
        }
    }
    resp = es.search(index='hd_index', query=query, size=num_docs)
    return resp
