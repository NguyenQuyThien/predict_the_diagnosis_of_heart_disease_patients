import sklearn

"""Warning: function normalization in scikit learn la chuan hoa theo rows (not columns) mặc định là axis=1, su dung MinMaxScaler, RobustScaler, StandardScaaler hon
            https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
    Phải chuẩn hóa trên 100% tập dữ liệu
"""


def rescaling_with_MinMaxScaler_sklearn(X):
    """

    :param X: X là np.array lưu data cần xử lý
    :return:  trả về data đc chuẩn hóa theo vector 0 -> 1
    """
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    # print(min_max_scaler.fit(X))
    return min_max_scaler.fit_transform(X)


def normalization_with_sklearn(X):
    """

    :param X:
    :return:
    """
    X_normalized_axis0 = sklearn.preprocessing.normalize(X, norm='l2', axis=0)
    return X_normalized_axis0
