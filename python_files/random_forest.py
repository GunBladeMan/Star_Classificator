import pickle

from imblearn.over_sampling import SMOTE


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler

data = pd.read_csv('D:/ProjectsVSCode/classificator_stars/csv/data_with_types2.csv')

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

data_from = data[features].loc[:220000]
data_to = data['present'].loc[:220000]

data_from_resampled, data_to_resampled = pipeline.fit_resample(data_from, data_to)

rand_forest = RandomForestClassifier(random_state=42, class_weight= {0:0.00001, 1:1})
rand_forest = rand_forest.fit(data_from_resampled, data_to_resampled)

with open(r"rand-forest-model-resampled.pickle", "wb") as fout:
    pickle.dump(rand_forest, fout)