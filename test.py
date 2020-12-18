import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleSDGothicNeoM00'

# import matplotlib.font_manager as fm
# for f in sorted([f.name for f in fm.fontManager.ttflist]):
#      print(f)

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


def run():
    df = pd.read_csv('src/2019-12-18~2020-12-18.csv', index_col='날짜')
    df.index = pd.to_datetime(df.index)

    # only 타입 == '지출'
    df = df[df['타입'] == '지출']

    # prerequisites
    categories = sorted(list(set(df['대분류'])))
    dates = []
    amounts = {
        category: []
        for category in categories
    }

    # make data
    for month, group in df.groupby(pd.Grouper(freq='W')):
        dates.append(month)

        for category in categories:
            if category == 'total':
                amounts[category].append(-group['금액'].sum())
            else:
                amounts[category].append(-group[group['대분류'] == category]['금액'].sum())

    # make as datafrmae
    for category in categories:
        print(category, amounts[category])
    df2 = pd.DataFrame(amounts, index=dates)

    # visualization
    fig, axes = plt.subplots(figsize=(20, 5))
    for category in categories:
        axes.plot(df2[category], label=category, linestyle='-', linewidth=1., marker='o', markersize=4)

    # axis settings
    axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))


    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run()
