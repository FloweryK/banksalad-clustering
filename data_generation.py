import numpy as np
import pandas as pd
from funcs import *
from config import *

# pandas max display options (only for code testing and monitoring)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def generate_from_clusters(load_path, mul):
    df_org = pd.read_csv(load_path, index_col=['날짜'], encoding='utf-8-sig')
    df_gen = []

    for label, group in df_org.groupby('label'):
        # drop label column for further calculation
        group = group.drop(columns=['label'])

        # get statistics
        n_samples = mul * len(group.index)
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

    # shuffle
    df_gen = df_gen.sample(frac=1)

    return df_gen


def generate_from_patterns(patterns, columns):
    df_gen = []

    for i, (pattern, size) in enumerate(patterns):
        samples = {category: np.zeros(size) for category in columns}

        for category, mean, sig in pattern:
            samples[category] = np.random.normal(mean, sig, size)
            samples[category] = np.clip(samples[category], a_min=0, a_max=None)
            samples[category] = np.clip(samples[category], a_min=mean - 2 * sig, a_max=mean + 2 * sig)

        samples = pd.DataFrame(
            data=samples,
            index=['pattern-%i-%i' % (i, j) for j in range(size)]
        )
        df_gen.append(samples)

    # concat all generated dataframe
    df_gen = pd.concat(df_gen)

    # shuffle
    df_gen = df_gen.sample(frac=1)

    return df_gen


if __name__ == '__main__':
    # df_gen = generate_from_clusters(load_path=SAVE_PATH, mul=5)
    patterns = [
        ([('식사', 100, 10), ('데이트', 50, 10)], 100),
        ([('식사', 10, 5), ('장보기', 90, 5)], 100),
        ([('식사', 30, 5), ('데이트', 30, 5), ('장보기', 30, 5)], 100),
    ]
    columns = ['식사', '데이트', '장보기']
    df_gen = generate_from_patterns(patterns, columns=columns)
    print(df_gen)

