import pandas as pd

# Инициализация списка для хранения объектов
objects = []

# Чтение файла
with open('asu.tsv', 'r') as f:
    lines = f.readlines()
    for index in range(len(lines)):
        if lines[index].startswith('#RESOURCE='):
            if lines[index + 9].startswith('_r'):
                line = lines[index + 12].split('\t')
                _object = {'present': 1, 'type': line[7]}
                objects.append(_object)
                index += 12
            else:
                index += 9
                _object = {'present': 0, 'type': None}
                objects.append(_object)

# # Добавление последнего объекта
# if _object is not None:
#     _object['present'] = present
#     _object['type'] = _type
#     objects.append(_object)

# Создание DataFrame из списка объектов
objects_df = pd.DataFrame(objects)

# Просмотр первых строк
print(objects_df)

data = pd.read_csv('D:/ProjectsVSCode/classificator_stars/csv/filtered_data_short.csv')
data["present"] = objects_df["present"]
data["type"] = objects_df["type"]

data.drop(data[data.nobs < 5].index, inplace=True)

data.to_csv('D:/ProjectsVSCode/classificator_stars/csv/data_with_types.csv', index=False, sep=',')
