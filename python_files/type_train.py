import pickle

from imblearn.over_sampling import SMOTE


import pandas as pd
from sklearn import linear_model, tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler

data = pd.read_csv('D:/ProjectsVSCode/classificator_stars/csv/types_defined.csv')

smote = SMOTE(
    sampling_strategy='auto',
    random_state=42,
    k_neighbors=5
)

pipeline = make_pipeline(
    RandomUnderSampler(),  # Сначала применяем андерсэмплинг
    SMOTE()                # Затем применяем SMOTE
)

features = ['angDist', 'RAJ2000', 'DEJ2000', 'errPosAng', 'nobs', 'mobs', 'B-V', 'e_B-V', 'gpmag', 'e_gpmag', 'rpmag', 'e_rpmag', 'ipmag', 'e_ipmag', 'nuv_mag', 'nuv_magerr', 'E_bv', 'nuv_flux', 'nuv_fluxerr']
d = {"ECLIPSING" : 0, 
     "RR_LYRAE" : 1,
     "DELTA_SCUTI_ETC" : 2,
     "LONG_PERIOD" : 3,
     "ROTATIONAL" : 4}

data['class'] = data['class'].map(d)

data_from = data[features].loc[:40000]
data_to = data['class'].loc[:40000]

data_from_resampled, data_to_resampled = smote.fit_resample(data_from, data_to)

# dtree = DecisionTreeClassifier(random_state=42)
# dtree = dtree.fit(data_from, data_to)

# knn = KNeighborsClassifier(n_neighbors=5)
# knn = knn.fit(data_from_resampled, data_to_resampled)

# regr = linear_model.LogisticRegression(random_state=42, penalty='l1', solver='liblinear', l1_ratio=0.6, n_jobs=-1)
# regr = regr.fit(data_from_resampled, data_to_resampled)

# oob_model = BaggingClassifier(n_estimators = 10, oob_score = True,random_state = 22, n_jobs=-1)
# oob_model = oob_model.fit(data_from_resampled, data_to_resampled)

rand_forest = RandomForestClassifier(random_state=42)
rand_forest = rand_forest.fit(data_from_resampled, data_to_resampled)

with open(r"rand-forest-model-type.pickle", "wb") as fout:
    pickle.dump(rand_forest, fout)