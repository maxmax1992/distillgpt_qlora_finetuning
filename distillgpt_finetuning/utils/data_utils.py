import re
import json
from typing import Dict, Tuple
import unicodedata
from ftfy import fix_text
from unidecode import unidecode
import pandas as pd

def get_train_df_and_append_target(json_data: Dict, labels_df: pd.DataFrame) -> pd.DataFrame:
    id_to_label_map = {}
    for i in range(len(labels_df)):
        item = labels_df.iloc[i, :]
        id, target = item.id, item.target
        id_to_label_map[id] = target

    for item in json_data:
        # print(id_to_label_map[item])
        item['target'] = id_to_label_map[item['id']]

    train_df = pd.DataFrame(json_data)
    return train_df


def decode_text(text):
    # Fix any mis-encoded characters
    try:
        text = fix_text(text)
        # Convert LaTeX to text
        # text = LatexNodes2Text().latex_to_text(text)
        text = text.encode("latex", errors="ignore").decode("utf-8")
        text = unicodedata.normalize('NFC', text)

        # Transliterate any remaining Unicode characters to ASCII if possible
        text = unidecode(text)
        text = text.replace('\n', ' ')
    except:
        pass
    return text
    
def add_sequential_info_todf(df_row):

    cols_to_normalize = ['submitter', 'authors', 'title', 'comments', 'abstract', 'journal-ref']
    for col in cols_to_normalize:
        df_row[col] = decode_text(df_row[col])

    all_authors = re.split(r',| and ', df_row.authors)
    all_authors = [x.strip() for x in all_authors]
    all_authors = [x for x in all_authors if len(x) > 0]
    top_5_authors = [author.strip() for author in all_authors[:5]]
    categories = df_row.categories.split(' ')
    df_row['categories'] = categories
    df_row['authors'] = top_5_authors
    df_row['n_versions'] = len(df_row['versions'])
    df_row['n_authors'] = len(df_row['authors_parsed'])
    df_row['n_categories'] = len(categories)

    return df_row

def delete_unnecessary_cols(curr_df):
    # keep the id to make final predictions
    # del curr_df['id']
    del curr_df['doi']
    del curr_df['versions']
    del curr_df['license']
    del curr_df['update_date']
    del curr_df['authors_parsed']
    # del df2['versions']
    return curr_df

def add_timedelta_features(curr_df):
    update_date = pd.to_datetime(curr_df['update_date'], format='%Y-%m-%d')
    reference_date = pd.to_datetime('2024-11-05')
    curr_df['days_since_update'] = (reference_date - update_date).dt.days

def preprocess_df(df):
    df2 = df.copy(deep=True)
    df2 = df2.fillna("")
    add_timedelta_features(df2)
    df2 = df2.apply(add_sequential_info_todf, axis=1)
    df2 = delete_unnecessary_cols(df2) 
    targets = df2.target - 1
    del df2['target']
    return df2, targets

def get_train_data(inputs_json_path: str, targets_csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open(inputs_json_path, 'r') as f:
        json_data = json.load(f)
    labels = pd.read_csv(targets_csv_path)

    train_df = get_train_df_and_append_target(json_data, labels)
    train_df, targets = preprocess_df(train_df)
    return train_df, targets

if __name__ == '__main__':
    # example usage of the preprocesing
    inputs_path ='./sample_data/sample_data.json'
    targets_path = "./sample_data/sample_targets.csv"

    train_df, targets = get_train_data(inputs_path, targets_path)
    print(train_df.head())
