import pickle

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score

with open(r"rand-forest-model-type.pickle", "rb") as fin:
    tree_loaded = pickle.load(fin)

data = pd.read_csv('D:/ProjectsVSCode/classificator_stars/csv/types_defined.csv')

features = ['angDist', 'RAJ2000', 'DEJ2000', 'errPosAng', 'nobs', 'mobs', 'B-V', 'e_B-V', 'gpmag', 'e_gpmag', 'rpmag', 'e_rpmag', 'ipmag', 'e_ipmag', 'nuv_mag', 'nuv_magerr', 'E_bv', 'nuv_flux', 'nuv_fluxerr']
d = {"ECLIPSING" : 0, 
     "RR_LYRAE" : 1,
     "DELTA_SCUTI_ETC" : 2,
     "LONG_PERIOD" : 3,
     "ROTATIONAL" : 4}

data['class'] = data['class'].map(d)

pred = list(tree_loaded.predict(data[features].loc[40000:]))
true = list(data['class'].loc[40000:])

cm = confusion_matrix(true, pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', annot_kws={'fontsize': 14})
plt.ylabel('Истинные значения')
plt.xlabel('Предсказанные значения')
plt.title('Матрица ошибок', pad=15)
plt.show()
