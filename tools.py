import pandas as pd

file = 'data/spam/spam.csv'

data = pd.read_csv(file,header=None)
print(data.head())
print(len(data))

file = 'data/GSD/gsd.csv'
with open(file) as f:
    print(len(f.readlines()))

