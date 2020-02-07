from sklearn.ensemble.forest import RandomForestClassifier
import numpy as np
import pickle
import matplotlib.pyplot as plt
import timeit
from sklearn.feature_selection import SelectFromModel

import Read_Data
import Score_Of_Algorithm

data = {}
XFeatureImportances = {}
for index in range(1, int(Read_Data.count_files("./folds")/2) + 1):
    data[f'X{index}_train'], data[f'y{index}_train']= Read_Data.read_data_in_folds_folder(f'./folds/patient{index}_train.csv')
    data[f'X{index}_test'] , data[f'y{index}_test'] = Read_Data.read_data_in_folds_folder(f'./folds/patient{index}_test.csv')
    # lấy các thuộc tính quan trọng nhất từ bộ dữ liệu (từ 6-13 thuộc tính)
    XFeatureImportances[f'X{index}Importances_train'] = data[f'X{index}_train'][:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    XFeatureImportances[f'X{index}Importances_test'] = data[f'X{index}_test'][:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
# Create a list of feature names
feat_labels = ['Age', 'Sex', 'Cp', 'Trestbps', 'Chol', 'Fbs', 'Restecg', 'Thalach', 'Exang', 'Oldpeak', 'Slope', 'Ca', 'Thal']
rf = RandomForestClassifier(n_estimators = 50,random_state = 1, max_depth = 5)

accuracy_score, precision_score, recall_score, f1_score, time = [], [], [], [], []
for index in range(1, int(len(data)/4) + 1):
    time_start = timeit.default_timer()
    # rf.fit(data[f'X{index}_train'], data[f'y{index}_train'])
    rf.fit(XFeatureImportances[f'X{index}Importances_train'], data[f'y{index}_train'])
    time_stop = timeit.default_timer()
    time.append(time_stop-time_start)
    print("Total time : %.2f ms" % (1000 * (time[index-1])))

    # # >>> Calculate feature importances, Random Forest Feature Importance Chart using Python
    # importances = rf.feature_importances_
    # # Print the name and the importance scores for each feature
    # for feature in zip(feat_labels, importances):
    #     print(feature)
    # # Sort feature importances in descending order
    # indices = np.argsort(importances)[::-1]
    # # Rearrange feature names so they match the sorted feature importances
    # names = [feat_labels[i] for i in indices]
    # # Barplot: Add bars
    # plt.bar(range(data[f'X{index}_train'].shape[1]), importances[indices])
    # # Add feature names as x-axis labels
    # plt.xticks(range(data[f'X{index}_train'].shape[1]), names, rotation=20, fontsize=8)
    # # Create plot title
    # plt.title(f'Độ quan trọng của các thuộc tính (Fold {index})')
    # # Show plot
    # plt.show()

    # Saving model to disk
    pickle.dump(rf, open(f'./models/Random_Forest_{index}.pkl', 'wb'))

    # >>> Đo độ chính xác voi 14/14 thuoc tinh
    # pred = rf.predict(data[f'X{index}_test'])
    # >>> Đo độ chính xác voi Important Features
    pred = rf.predict(XFeatureImportances[f'X{index}Importances_test'])

    accuracy_score.append(Score_Of_Algorithm.accuracy_can_modify(data[f'y{index}_test'], pred))

    x, y = Score_Of_Algorithm.precision_recall_can_modify(data[f'y{index}_test'], pred)
    precision_score.append(x)
    recall_score.append(y)

    f1_score.append(Score_Of_Algorithm.f1_score_can_modify(data[f'y{index}_test'], pred))
    print("Accuracy score:", accuracy_score[index - 1])
    print("Precision and Recall score:", precision_score[index - 1], recall_score[index - 1])
    print("F1 score:", f1_score[index - 1])

    print("===================================================================")
# print avg_score
print("avg total time : %.2f ms" % (1000 * (np.sum(time) / (int(len(data) / 4)))))
print("avg of accuracy score: ", np.sum(accuracy_score) / (int(len(data) / 4)))
print("avg of precision score: ", np.sum(precision_score) / (int(len(data) / 4)))
print("avg of recall score: ", np.sum(recall_score) / (int(len(data) / 4)))
print("avg of F1 score: ", np.sum(f1_score) / (int(len(data) / 4)))
data = None
