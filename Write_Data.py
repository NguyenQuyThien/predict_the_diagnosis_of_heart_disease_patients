import K_Fold

import csv
import re

def write_data(file_name, data):
    """
    write datas to files in the folds folder. ex: patient1_train.csv, patient1_test.csv, patient1_train.csv,...
    :param file_name: tên file ex: patient1_train
    :param data: wite this data to file
    :return:
    """
    with open(f'{file_name}.csv', mode='w') as the_file:
        writer_data = csv.writer(the_file, delimiter=',')
        # data_writer.writerow(['Age', 'Sex', 'Cp', 'Trestbps', 'Chol', 'Fbs', 'Restecg', 'Thalach', 'Exang', 'Oldpeak', 'Slope', 'Ca', 'Thal', 'Result'])
        # print(file_name)
        # if re.search('[y][\d][_]', file_name) is not None:
        #     writer_data.writerow(data)
        # else:
        #     for i in data:
        #         writer_data.writerow(i)
        for i in data:
            writer_data.writerow(i)

if __name__ == '__main__':
    for value in K_Fold.data:
        # print(value)
        write_data(f'./folds/{value}', K_Fold.data[f'{value}'])

    data = None
"""
    Chia K_fold co xao tron sao cho chay thuat toan ma co F1_score cang gan nhau cang tot
"""