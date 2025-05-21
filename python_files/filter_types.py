import csv
import pandas as pd

data = pd.read_csv('D:/ProjectsVSCode/classificator_stars/types_train.csv')
for x in data.index:
    if data.loc[x, "present"] == 0 :
        data.drop(x, inplace=True)

data.to_csv('D:/ProjectsVSCode/classificator_stars/types_train_clean.csv', index=False, sep=',')
