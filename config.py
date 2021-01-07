# DATA LOAD & GENERATE
LOAD_PATH = 'src/2019-12-18~2020-12-18.xlsx'
SAVE_PATH = 'df_label.csv'
SAVE = False
GENERATE = True
PATTERNS = [
    ([('식사', 100, 10), ('데이트', 50, 10)], 100),
    ([('식사', 10, 5), ('장보기', 90, 5)], 100),
    ([('식사', 30, 5), ('데이트', 30, 5), ('장보기', 30, 5)], 100)
]

# PREPROCESSING
NORM = True

# CLUSTERING
METRIC = 'cosine'
TRIALS = 1
