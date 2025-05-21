import pickle

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score

with open(r"knn-model-resampled.pickle", "rb") as fin:
    tree_loaded = pickle.load(fin)

data = pd.read_csv('D:/ProjectsVSCode/classificator_stars/csv/data_with_types2.csv')
features = ['angDist', 'RAJ2000', 'DEJ2000', 'errPosAng', 'nobs', 'mobs', 'B-V', 'e_B-V', 'gpmag', 'e_gpmag', 'rpmag', 'e_rpmag', 'ipmag', 'e_ipmag', 'nuv_mag', 'nuv_magerr', 'E_bv', 'nuv_flux', 'nuv_fluxerr']

pred = list(tree_loaded.predict(data[features].loc[220000:]))
true = list(data['present'].loc[220000:])

cm = confusion_matrix(true, pred)

Accuracy = metrics.accuracy_score(true, pred)
Precision = metrics.precision_score(true, pred)
Sensitivity_recall = metrics.recall_score(true, pred)
Specificity = metrics.recall_score(true, pred, pos_label=0)
F1_score = metrics.f1_score(true, pred)

print("Accuracy: " + str(Accuracy))
print("Precision: " + str(Precision))
print("Recall: " + str(Sensitivity_recall))
print("Specificity: " + str(Specificity))
print("F1 score: " + str(F1_score))
print(f'AUC score: {roc_auc_score(pred, true)}')

sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', annot_kws={'fontsize': 14})
plt.ylabel('Истинные значения')
plt.xlabel('Предсказанные значения')
plt.title('Матрица ошибок', pad=15)
plt.show()
