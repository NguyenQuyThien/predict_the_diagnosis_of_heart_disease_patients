import sklearn.metrics
import sklearn.neighbors
import numpy as np
import pickle

import Read_Data
import Score_Of_Algorithm


data = {}
for index in range(1, int(Read_Data.count_files("./folds")/2) + 1):
    data[f'X{index}_train'], data[f'y{index}_train']= Read_Data.read_data_in_folds_folder(f'./folds/patient{index}_train.csv')
    data[f'X{index}_test'] , data[f'y{index}_test'] = Read_Data.read_data_in_folds_folder(f'./folds/patient{index}_test.csv')
# print(data["X1_train"], "\n", data["y1_test"])
accuracy_score, precision_score, recall_score, f1_score = [], [], [], []
neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3, p=2)

for index in range(1, int(len(data)/4) + 1):
    neigh.fit(data[f'X{index}_train'], data[f'y{index}_train'])
    # Saving model to disk
    pickle.dump(neigh, open(f'./models/KNN_{index}.pkl','wb'))

    pred = neigh.predict(data[f'X{index}_test'])

    # print(sklearn.metrics.classification_report(data[f'y{index}_test'], pred))
    accuracy_score.append(Score_Of_Algorithm.accuracy_can_modify(data[f'y{index}_test'], pred))

    x, y = Score_Of_Algorithm.precision_recall_can_modify(data[f'y{index}_test'], pred)
    precision_score.append(x)
    recall_score.append(y)

    f1_score.append(Score_Of_Algorithm.f1_score_can_modify(data[f'y{index}_test'], pred))
    print("Accuracy score:", accuracy_score[index-1])
    print("Precision and Recall score:", precision_score[index-1], recall_score[index-1] )
    print("F1 score:", f1_score[index-1])

    # print ("Print results for 20 test data points:")
    # print ("Predicted labels: ", y1_pred[20:40])
    # print ("Ground truth    : ", data['y1_test'][20:40])
    print("===================================================================")
# print avg_score
print("avg of accuracy score: ", np.sum(accuracy_score)/(int(len(data)/4)))
print("avg of precision score: ", np.sum(precision_score)/(int(len(data)/4)))
print("avg of recall score: ", np.sum(recall_score)/(int(len(data)/4)))
print("avg of F1 score: ", np.sum(f1_score)/(int(len(data)/4)))
data = None

"""

>>> Arttificial Intelligence with Python - Que
        Once the model has been created, we can save it into a file so that we can use it later. Python
            provides a nice module called pickle that enables us to do this:
            
# Model persistence
    output_model_file = 'model.pkl'
# Save the model
    with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)\
    
        Let's load the model from the file on the disk and perform prediction:
# Load the model
with open(output_model_file, 'rb') as f:
regressor_model = pickle.load(f)
# Perform prediction on test data
y_test_pred_new = regressor_model.predict(X_test)
print("\nNew mean absolute error =", round(sm.mean_absolute_error(y_test,
y_test_pred_new), 2))

=========================================================
>>> Let's shuffle the data so that we don't bias our analysis:
# Shuffle the data
X, y = shuffle(data.data, data.target, random_state=7)

"""
