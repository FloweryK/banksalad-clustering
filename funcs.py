import pandas as pd
from openpyxl import load_workbook
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def load_banksalad_as_df(path):
    # open xlsx worksheet
    wb = load_workbook(path)
    ws = wb['가계부 내역']

    # convert worksheet into dataframe
    raw = pd.DataFrame(ws.values)
    raw.columns = raw.loc[0].tolist()
    raw = raw.drop(index=[0])
    raw.index = raw['날짜']
    raw = raw.drop(columns=['날짜'])

    # extract target types
    # TODO: currently using hard-coded type only
    raw = raw[raw['타입'] == '지출']

    # make categorized daily outcome
    # TODO: currently using hard-coded high-hierarchy category only
    df = pd.DataFrame(columns=set(raw['대분류']))

    # TODO: currently using hard-coded grouping options
    for date, group in raw.groupby(pd.Grouper(freq='W')):
        df.loc[date] = {category: abs(group2['금액'].sum()) for category, group2 in group.groupby('대분류')}

    # beautify dataframe
    df = df.fillna(0)
    df = df.sort_index(axis=0)
    df = df.sort_index(axis=1, key=lambda x: df[x].sum(), ascending=False)
    df.index = df.index.strftime('%Y-%m-%d')
    df.index.name = '날짜'

    return df


def normalize(df, norm, mean):
    if norm:
        # TODO: more normalize options (ex. axis=0 dividing, L2 norm)
        df = df.div(df.sum(axis=1), axis=0)
    if mean:
        df -= df.mean(axis=0)

    return df


def convert_metric(df, measure):
    if measure == 'cosine':
        return 1 - cosine_similarity(df, df)
    elif measure == 'euclidean':
        return euclidean_distances(df, df)
    else:
        raise KeyError
