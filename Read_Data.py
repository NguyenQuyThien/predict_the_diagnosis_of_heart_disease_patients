import numpy as np
import sklearn.impute

import csv
import os
import re
import Normalization

""" All missing attribute values are denoted either by ”?” or by ”*”:
            lost values will be denoted by ”?”,
            ”do not care” conditions will be denoted by ”*”. 
    Missing values problem: sklearn provides a SimpleImputer. 
            The four main strategies are mean, most_frequent, median and constant (don’t forget to set the fill_value parameter).

    Other way load data from file:
        input_file = 'data_multivar_nb.txt'

        data = np.loadtxt(input_file, delimiter=',')
        X, y = data[:, :-1], data[:, -1]
"""


def read_data_in_folds_folder(input_file):
    """
    use to load data in the folds folder
    :param input_file: name of the fold file
    :return: data X and result y
    """
    data = np.loadtxt(input_file, delimiter=',')
    return data[:, :-1], data[:, -1]
    # if re.search('[y][\d][_]', input_file) is not None:
    #     return np.array(data[:])
    # else:
    #     return np.array(data[:, :])


def count_files(dir):
    """
    count number of the file in the fold folder
    :param dir: path to folder
    :return: num of file
    """
    return len([1 for x in list(os.scandir(dir)) if x.is_file()])


def remove_missing_values(X):
    """

    :param X: data X. Nên là data có cả thuộc tính cùng result
    :return: data with remove missing values
    """
    temp_X = []
    missing_values = 'nan'
    for row in X:
        if (missing_values in str(row)):
            continue
        temp_X.append(row)
    return temp_X


def missing_values_with_most_frequent_values(X):
    imp_mean = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    return imp_mean.fit_transform(X)


def missing_values_with_the_mean_values(X):
    imp_mean = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    return imp_mean.fit_transform(X)


with open('processed_file.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    patient0, patient1 = [], []

    for row in csv_reader:
        # print(line_count)
        if line_count == 0:
            line_count += 1
            continue
        temp = np.array([(float(row[0]) if not (row[0] == '-9') else np.NaN), \
                         (float(row[1]) if not (row[1] == '-9') else np.NaN), \
                         (float(row[2]) if not (row[2] == '-9') else np.NaN), \
                         (float(row[3]) if not (row[3] == '-9') else np.NaN), \
                         (float(row[4]) if not (row[4] == '-9') else np.NaN), \
                         (float(row[5]) if not (row[5] == '-9') else np.NaN), \
                         (float(row[6]) if not (row[6] == '-9') else np.NaN), \
                         (float(row[7]) if not (row[7] == '-9') else np.NaN), \
                         (float(row[8]) if not (row[8] == '-9') else np.NaN), \
                         (float(row[9]) if not (row[9] == '-9') else np.NaN), \
                         (float(row[10]) if not (row[10] == '-9') else np.NaN), \
                         (float(row[11]) if not (row[11] == '-9') else np.NaN), \
                         (float(row[12]) if not (row[12] == '-9') else np.NaN), \
                         (float(row[13]) if (float(row[13]) == 0) else 1)])

        # y.append(float(row[13]))
        if (float(row[13]) == 0):
            patient0.append(temp)
        else:
            patient1.append(temp)

        # if line_count == 10: break #                        *********** gioi han 10 mau
        line_count += 1

patient = np.append(patient0, patient1, axis=0)

# patient_with_most_frequent = missing_values_with_most_frequent_values(patient)
patient_with_the_mean_values = missing_values_with_the_mean_values(patient)
# patient_with_remove_missing_values = remove_missing_values(patient)

patient_scales = Normalization.rescaling_with_MinMaxScaler_sklearn(patient_with_the_mean_values)

patient, patient_with_most_frequent, patient_with_the_mean_values, patient_with_remove_missing_values = None, None, None, None