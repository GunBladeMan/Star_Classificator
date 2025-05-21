import pandas as pd

data = pd.read_csv('D:/ProjectsVSCode/classificator_stars/csv/data2.csv')
features = ['angDist', 'RAJ2000', 'DEJ2000', 'errHalfMaj', 'errHalfMin', 'errPosAng', 'field', 'nobs', 'mobs', 'B-V', 'e_B-V', 'Vmag', 'e_Vmag', 'Bmag', 'e_Bmag', 'gpmag', 'e_gpmag', 'rpmag', 'e_rpmag', 'ipmag', 'e_ipmag', 'ra', 'dec', 'nuv_mag', 'nuv_magerr', 'E_bv', 'nuv_flux', 'nuv_fluxerr']
data.dropna(inplace=True, subset=features)

cord = {
        'RAJ2000' : data["RAJ2000"],
        'DEJ2000' : data["DEJ2000"]
}

cdata = pd.DataFrame(cord)
cdata.to_csv('D:/ProjectsVSCode/classificator_stars/test/cord2.csv', index=False, sep=',')
nd = pd.DataFrame(data[features])
nd.to_csv('D:/ProjectsVSCode/classificator_stars/test/filtered_data2.csv', index=False, sep=',')