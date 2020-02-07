import sklearn.model_selection
import numpy as np

import Read_Data
# tách data thành data bệnh nhân âm tính, bệnh nhân dương tính
tempPatient0, tempPatient1 = [], []
for row in Read_Data.patient_scales:
    if row[13] == 0:
        tempPatient0.append(row)
    else:
        tempPatient1.append(row)
tempPatient0 = np.array(tempPatient0)
tempPatient1 = np.array(tempPatient1)


kf = sklearn.model_selection.KFold(n_splits=5, random_state=None, shuffle=True)
"""
    1. kfold patient có nhãn 0,1 thành 5 folds
    2. merge patient0 and patient1 => data train và data test (có tỷ lệ nhãn 0, 1 đều nhau)
"""
data0, data1 = {}, {}
index = 1
for train_index, test_index in kf.split(tempPatient0):
    # print("TRAIN:", train_index, "TEST:", test_index)
    data0[f'patient0_{index}_train'], data0[f'patient0_{index}_test'] = tempPatient0[train_index], tempPatient0[test_index]
    index += 1

index = 1
for train_index, test_index in kf.split(tempPatient1):
    # print("TRAIN:", train_index, "TEST:", test_index)
    data1[f'patient1_{index}_train'], data1[f'patient1_{index}_test'] = tempPatient1[train_index], tempPatient1[test_index]
    index += 1

# merge patient0 and patient1 => data (có tỷ lệ result 0, 1 đều nhau)
data = {}
index = 1
for i in range(1, 6):
    data[f'patient{index}_train'], data[f'patient{index}_test'] = \
        np.append(data0[f'patient0_{i}_train'], data1[f'patient1_{i}_train'], axis=0), \
        np.append(data0[f'patient0_{i}_test'], data1[f'patient1_{i}_test'], axis=0)
    index += 1
print(data.keys())
print()
"""
    data: X la tap tham so dau vao, y la tap ket qua
        X1_train, y1_train dung de train thuat toan
        X1_test , y1_test dung de test thuat toan        
"""

""" The code: y = y.reshape(y.shape [1:]) fix error in KNN_Algorithm (.fit(data['X1_train'], data['y1_train']))
        https://datascience.stackexchange.com/questions/20199/train-test-split-error-found-input-variables-with-inconsistent-numbers-of-sam
"""
