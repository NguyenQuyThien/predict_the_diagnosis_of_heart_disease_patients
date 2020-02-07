import sklearn.svm
import sklearn.metrics
import numpy as np
import pickle

import Read_Data
import Score_Of_Algorithm


data = {}
for index in range(1, int(Read_Data.count_files("./folds")/2) + 1):
    data[f'X{index}_train'], data[f'y{index}_train']= Read_Data.read_data_in_folds_folder(f'./folds/patient{index}_train.csv')
    data[f'X{index}_test'] , data[f'y{index}_test'] = Read_Data.read_data_in_folds_folder(f'./folds/patient{index}_test.csv')

clf = sklearn.svm.SVC(kernel = 'linear', C = 1e5) # just a big number

# clf.fit(data["X1_train"], data["y1_train"])

# w = clf.coef_
# b = clf.intercept_
# print('w = ', w)
# print('b = ', b)
accuracy_score, precision_score, recall_score, f1_score = [], [], [], []
for index in range(1, int(len(data)/4) + 1):
    clf.fit(data[f'X{index}_train'], data[f'y{index}_train'])
    # Saving model to disk
    pickle.dump(clf, open(f'./models/SVM_{index}.pkl', 'wb'))

    pred = clf.predict(data[f'X{index}_test'])

    print(sklearn.metrics.classification_report(data[f'y{index}_test'], pred))
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
print("avg of accuracy score: ", np.sum(accuracy_score)/(int(len(data)/4)))
print("avg of precision score: ", np.sum(precision_score)/(int(len(data)/4)))
print("avg of recall score: ", np.sum(recall_score)/(int(len(data)/4)))
print("avg of F1 score: ", np.sum(f1_score)/(int(len(data)/4)))
data = None
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(data["y1_test"],pred))
# print(classification_report(data["y1_test"],pred))