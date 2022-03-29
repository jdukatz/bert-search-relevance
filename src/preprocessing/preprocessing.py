import os
import pandas as pd

from preprocessing.constants import DATA_DIRECTORY, TRAIN_FILE, ATTRIBUTES_FILE, DESCRIPTIONS_FILE


def join_attributes(df):
    name = df['name']
    val = df['value']
    if 'Bullet' in name:
        name = ''
    return name + ' ' + val


def convert_raw_data_to_product_records():
    print('Loading and transforming data...')
    train_data = pd.read_csv(os.path.join('..', DATA_DIRECTORY, TRAIN_FILE), encoding='ISO-8859-1')
    descriptions = pd.read_csv(os.path.join('..', DATA_DIRECTORY, DESCRIPTIONS_FILE))
    attributes = pd.read_csv(os.path.join('..', DATA_DIRECTORY, ATTRIBUTES_FILE))

    num_products = len(train_data['product_uid'].unique())
    print(f'{num_products} products found in train data')
    # convert to int since product id is a float here for some reason
    attributes.dropna(inplace=True)
    attributes['product_uid'] = attributes['product_uid'].astype(int)

    unique_products = train_data[['product_uid', 'product_title']] \
        .drop_duplicates('product_uid', keep='first') \
        .join(descriptions.set_index('product_uid'), on='product_uid')
    attributes['name_value_joined'] = attributes.apply(join_attributes, axis=1)
    attributes_by_product = attributes.groupby('product_uid').apply(lambda group: '\n'.join(group['name_value_joined']))

    product_records = unique_products.to_dict(orient='records')
    for json_record in product_records:
        uid = json_record['product_uid']
        try:
            extra_attributes = attributes_by_product[uid]
            json_record['attributes'] = extra_attributes
        except KeyError:
            print(f'No extra attributes found for product uid: {uid}')
    print('Done.')
    return product_records
