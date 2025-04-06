import csv
import pandas as pd

data = pd.read_csv('D:/ProjectsVSCode/classificator_stars/csv/data.csv')
features = ['nobs', 'B-V', 'e_B-V', 'fuv_mag', 'nuv_mag', 'ipmag', 'gpmag', 'rpmag']
data.dropna(inplace=True, subset=features)
ndict = {'angDist' : data["angDist"],
         'RAJ2000' : data["RAJ2000"],
         'DEJ2000' : data["DEJ2000"],
         'nobs' : data["nobs"],
         'B-V' : data["B-V"],
         'e_B-V' : data["e_B-V"],
         'fuv_mag' : data["fuv_mag"],
         'nuv_mag' : data["nuv_mag"],
         'ipmag' : data["ipmag"],
         'gpmag' : data["gpmag"],
         'rpmag' : data["rpmag"]}
cord = {
        'RAJ2000' : data["RAJ2000"],
        'DEJ2000' : data["DEJ2000"]
}
cdata = pd.DataFrame(cord)
cdata.to_csv('D:/ProjectsVSCode/classificator_stars/test/cord.csv', index=False, sep=',')
nd = pd.DataFrame(ndict)
nd.to_csv('D:/ProjectsVSCode/classificator_stars/test/filtered_data_short.csv', index=False, sep=',')
# with open('D:/ProjectsVSCode/classificator_stars/filtered_data.csv', 'w', newline='') as csvfile:
#     result = csv.writer(csvfile, delimiter=';')
#     result.writerows(data)