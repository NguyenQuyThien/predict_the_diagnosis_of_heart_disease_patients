import numpy as np


def accuracy_can_modify(y_true, y_pred):
    """
    Accuracy (độ chính xác): tỉ lệ giữa số điểm được dự đoán đúng và tổng số điểm trong tập dữ liệu kiểm thử.
    :param y_true:
    :param y_pred:
    :return:
    """
    # one way:
    # return (100.0 * (data['y1_test'] == y1_pred).sum() / data['X1_test'].shape[0])
    # other way:
    correct = np.sum(y_true == y_pred)
    return float(correct)/y_true.shape[0]


def precision_recall_can_modify(y_true, y_pred):
    """
    Precision tỉ lệ số điểm true positive trong số những điểm được phân loại là positive (TP + FP).
    Recall tỉ lệ số điểm true positive trong số những điểm thực sự là positive (TP + FN).
    :param y_true: vector chuan cua data
    :param y_pred: vector du doan cua algorithm
    :return: gia tri cua precision, recall
    """
    truePositive = positive = falseNegative = 0
    for index in range(y_true.size):
        if (y_true[index] == 1):
            if (y_pred[index] == 1):    truePositive += 1
            if (y_pred[index] == 0):    falseNegative += 1
        if (y_pred[index] == 1):   positive += 1
    return ( float(truePositive/(positive)), float(truePositive/(truePositive + falseNegative)) )


def f1_score_can_modify(y_true, y_pred):
    """
    F1-score có giá trị nằm trong nửa khoảng (0,1]. F1 càng cao, bộ phân lớp (class) càng tốt.
    F1 = 2 (precision * recall) / (precision + recall)
    :param y_true:
    :param y_pred:
    :return:
    """
    temp = precision_recall_can_modify(y_true, y_pred)
    return 2*(temp[0] * temp[1]) / (temp[0] + temp[1])


# y_true = np.array([0, 0, 1, 0, 1, 1, 1, 0, 0, 1])
# y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 0, 1, 1])
#
# print('accuracy = ', accuracy(y_true, y_pred))
# print('precision_recall = ', precision_recall(y_true, y_pred))
# print(f1_score(y_true, y_pred))