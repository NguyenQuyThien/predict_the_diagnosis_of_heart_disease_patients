from sklearn.ensemble.forest import RandomForestClassifier
import numpy as np
import pickle
import timeit
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel

import Normalization
import Read_Data

# đọc dữ liệu vs thư viện pandas
df = pd.read_csv("processed_file.csv")
# thay biến class mục tiêu 0,1,2,3,4 thành 0,1
df['Result'] = df['Result'].replace([1, 2, 3, 4], 1)
# thay missing value bằng NaN rồi chuẩn hóa bằng 1 trong 3 cách rồi lưu vào patient_scales
df[df==-9] = np.nan
patient = df.values
df = None
# patient_with_most_frequent = missing_values_with_most_frequent_values(patient)
patient_with_the_mean_values = Read_Data.missing_values_with_the_mean_values(patient)
# patient_with_remove_missing_values = remove_missing_values(patient)
patient_scales = Normalization.rescaling_with_MinMaxScaler_sklearn(patient_with_the_mean_values)
patient, patient_with_most_frequent, patient_with_the_mean_values, patient_with_remove_missing_values = None, None, None, None

data = {}
XFeatureImportances = {}
data[f'X_train'] = patient_scales[:, :13]
XFeatureImportances[f'XImportances_train'] = data[f'X_train'][:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

data[f'y_train'] = np.array(patient_scales[:, 13:]).ravel()

patient_scales = None

# Create a list of feature names
feat_labels = ['Age', 'Sex', 'Cp', 'Trestbps', 'Chol', 'Fbs', 'Restecg', 'Thalach', 'Exang', 'Oldpeak', 'Slope', 'Ca', 'Thal']
rf = RandomForestClassifier(n_estimators = 50,random_state = 1, max_depth = 5)

time_start = timeit.default_timer()
# >>> Train 14/14 thuoc tinh
# rf.fit(data[f'X_train'], data[f'y_train'])
# >>> Train with Feature Importances
rf.fit(XFeatureImportances[f'XImportances_train'], data[f'y_train'])
time_stop = timeit.default_timer()
print("Total time : %.2f ms" % (1000 * (time_stop - time_start)))

print("Finish training 100% data with Random Forest Model")
# >>> Test
# pred = rf.predict(XFeatureImportances[f'XImportances_train'])
# print(pred)

# >>> Calculate feature importances, Random Forest Feature Importance Chart using Python
importances = rf.feature_importances_
# Print the name and the importance scores for each feature
for feature in zip(feat_labels, importances):
    print(feature)
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [feat_labels[i] for i in indices]
# Barplot: Add bars
plt.bar(range(data[f'X_train'].shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(data[f'X_train'].shape[1]), names, rotation=20, fontsize=8)
# Create plot title
plt.title(f'Độ quan trọng của các thuộc tính')
# Show plot
plt.show()

# Saving model to disk
pickle.dump(rf, open(f'./models/Model_Random_Forest.pkl', 'wb'))

data, XFeatureImportances = None, None

# Danh sách thứ tự các thuộc tính (6-13) quan trọng
# [2, 4, 7, 8, 9, 10]
# [0, 2, 4, 7, 8, 9, 10]
# [0, 2, 4, 7, 8, 9, 10, 12]
# [0, 2, 4, 7, 8, 9, 10, 11, 12]
# [0, 1, 2, 4, 7, 8, 9, 10, 11, 12]
# [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12]
# [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]