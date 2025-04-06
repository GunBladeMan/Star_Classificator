import pickle

import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('D:/ProjectsVSCode/classificator_stars/csv/data_with_types.csv')

# features = ['angDist', 'RAJ2000', 'DEJ2000', 'B-V', 'e_B-V', 'fuv_mag', 'nuv_mag', 'ipmag', 'gpmag', 'rpmag']
features = ['angDist', 'RAJ2000', 'DEJ2000', 'B-V', 'e_B-V', 'ipmag', 'gpmag', 'rpmag']

data_from = data[features]
data_to = data['present']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(data_from, data_to)

tree.plot_tree(dtree, feature_names=features)

with open(r"decision-tree-model.pickle", "wb") as fout:
    pickle.dump(dtree, fout)