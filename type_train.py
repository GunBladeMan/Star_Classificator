import pickle

import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('D:/ProjectsVSCode/classificator_stars/csv/types_classified.csv')

features = ['angDist', 'RAJ2000', 'DEJ2000', 'B-V', 'e_B-V', 'fuv_mag', 'nuv_mag', 'ipmag', 'gpmag', 'rpmag']
# features = ['angDist', 'RAJ2000', 'DEJ2000', 'B-V', 'e_B-V', 'ipmag', 'gpmag', 'rpmag']
d = {"ECLIPSING" : 0, 
     "CEPHEIDS" : 1, 
     "RR_LYRAE" : 2,
     "DELTA_SCUTI_ETC" : 3,
     "LONG_PERIOD" : 4,
     "ROTATIONAL" : 5,
     "YSO/ERUPTIVE" : 6,
     "CATACLYSMIC" : 7,
     "EMISSION_WR" : 8,
     "UNKNOWN" : 9}

data_from = data[features].loc[:8000]
data['class'] = data['class'].map(d)
data_to = data['class'].loc[:8000]

dtree = DecisionTreeClassifier()
dtree = dtree.fit(data_from, data_to)

tree.plot_tree(dtree, feature_names=features)

with open(r"decision-tree-model-type.pickle", "wb") as fout:
    pickle.dump(dtree, fout)