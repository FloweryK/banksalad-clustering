import numpy as np
import pandas as pd
from funcs import *
from config import *

# pandas max display options (only for code testing and monitoring)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)


def run(df_org_path):
    df_gen = generate_from_clusters(df_org_path, n_samples=100)
    print(df_gen)


def generate_from_clusters(df_org_path, n_samples):
    df_org = pd.read_csv(df_org_path, index_col=['날짜'], encoding='utf-8-sig')
    df_gen = []

    for label, group in df_org.groupby('label'):
        # drop label column for further calculation
        group = group.drop(columns=['label'])

        # get statistics
        mean = group[group != 0].mean(axis=0, skipna=True).fillna(0)
        std = group[group != 0].std(axis=0, skipna=True).fillna(0.1)
        nonzero_prob = group.astype(bool).sum(axis=0) / len(group.index)

        samples = pd.DataFrame(
            data={category: (np.random.uniform(0, 1, size=n_samples) <= nonzero_prob[category]) * abs(np.random.normal(loc=mean[category], scale=std[category], size=n_samples)) for category in group.columns},
            index=['cluster-%i-%i' % (label, i) for i in range(n_samples)]
        )
        samples.index.name = '날짜'
        df_gen.append(samples)

    # concat all generated dataframe
    df_gen = pd.concat(df_gen)

    return df_gen


if __name__ == '__main__':
    run(df_org_path=DF_ORG_PATH)
