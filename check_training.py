import pickle

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix

with open(r"decision-tree-model-type.pickle", "rb") as fin:
    tree_loaded = pickle.load(fin)

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

pred = list(tree_loaded.predict(data[features].loc[8000:]))
true = list(data['class'].map(d).loc[8000:])

cm = confusion_matrix(true, pred)

# Accuracy = metrics.accuracy_score(true, pred)
# Precision = metrics.precision_score(true, pred)
# Sensitivity_recall = metrics.recall_score(true, pred)
# Specificity = metrics.recall_score(true, pred, pos_label=0)
# F1_score = metrics.f1_score(true, pred)

# print("Accuracy: " + str(Accuracy))
# print("Precision: " + str(Precision))
# print("Recall: " + str(Sensitivity_recall))
# print("Specificity: " + str(Specificity))
# print("F1 score: " + str(F1_score))

sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', annot_kws={'fontsize': 14})
plt.ylabel('Истинные значения')
plt.xlabel('Предсказанные значения')
plt.title('Матрица ошибок', pad=15)
plt.show()
