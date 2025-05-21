import pickle

from imblearn.over_sampling import SMOTE


import pandas as pd
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn import linear_model

data = pd.read_csv('D:/ProjectsVSCode/classificator_stars/csv/data_with_types2.csv')

features = ['angDist', 'RAJ2000', 'DEJ2000', 'errPosAng', 'nobs', 'mobs', 'B-V', 'e_B-V', 'gpmag', 'e_gpmag', 'rpmag', 'e_rpmag', 'ipmag', 'e_ipmag', 'nuv_mag', 'nuv_magerr', 'E_bv', 'nuv_flux', 'nuv_fluxerr']

smote = SMOTE(
    sampling_strategy='auto',
    random_state=42,
    k_neighbors=5
)

pipeline = make_pipeline(
    RandomUnderSampler(),  # Сначала применяем андерсэмплинг
    SMOTE()                # Затем применяем SMOTE
)

data_from = data[features].loc[:220000]
data_to = data['present'].loc[:220000]

data_from_resampled, data_to_resampled = pipeline.fit_resample(data_from, data_to)

regr = linear_model.LogisticRegression(random_state=42, max_iter=1000000000)
regr = regr.fit(data_from_resampled, data_to_resampled)

with open(r"logistic-regr-model-resampled.pickle", "wb") as fout:
    pickle.dump(regr, fout)

