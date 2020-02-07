import sklearn.metrics
import sklearn.naive_bayes
import numpy as np
import pickle

import Read_Data
import Score_Of_Algorithm

data = {}
for index in range(1, int(Read_Data.count_files("./folds")/2) + 1):
    data[f'X{index}_train'], data[f'y{index}_train']= Read_Data.read_data_in_folds_folder(f'./folds/patient{index}_train.csv')
    data[f'X{index}_test'] , data[f'y{index}_test'] = Read_Data.read_data_in_folds_folder(f'./folds/patient{index}_test.csv')

# Create Naïve Bayes classifier
classifier = sklearn.naive_bayes.GaussianNB()
"""
# Compute accuracy
accuracy = 100.0 * (data['y1_test'] == y1_pred).sum() / data['X1_test'].shape[0]
print("Accuracy of Naïve Bayes classifier =", round(accuracy, 2), "%")
"""
accuracy_score, precision_score, recall_score, f1_score = [], [], [], []
for index in range(1, int(len(data)/4) + 1):
    classifier.fit(data[f'X{index}_train'], data[f'y{index}_train'])
    # Saving model to disk
    pickle.dump(classifier, open(f'./models/Naive_Bayes_{index}.pkl', 'wb'))

    pred = classifier.predict(data[f'X{index}_test'])

    # print(sklearn.metrics.classification_report(data[f'y{index}_test'], pred))
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
"""
from Utilities import visualize_classifier
# Visualize the performance of the classifier
# visualize_classifier(classifier, data['X1_test'], data['y1_test'])
print('=================================================')
"""