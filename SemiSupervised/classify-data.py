import pandas as pd
import numpy as np

"""with open('../data/training.data', 'r') as f:
    data = f.readlines()
    for i, d in enumerate(data):
        ts = 0
        for c in d:
            if c == '\t':
                ts += 1
        if ts > 1:
            print('more ts in', i)
"""

data = pd.read_csv('../data/training.data', sep='\t', error_bad_lines=False, header=None)
classes = data.iloc[:, 1].unique()
numbers = dict()
for i, c in enumerate(classes):
    numbers[c] = i
for i, d in enumerate(data.iloc[:, 1]):
    data.iloc[i, 1] = numbers[d]
data.to_csv('../data/training.datacla')
