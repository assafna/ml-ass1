import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

MIN_SAMPLES_TO_SPLIT = 10
MAX_DEPTH = 5
K_FOLD_SPLITS = 10


def read_and_process_data(_file):
    _data = pd.read_csv(_file, header=None)

    # clean
    _data = _data.replace('?', np.nan)
    _data = _data.dropna()

    # split to x and y
    _x, _y = _data.iloc[:, :-1], _data.iloc[:, -1]

    # add prefix to each nominal column to avoid duplicated columns
    for _column_index in range(_x.shape[1]):
        _column_values = _x[_column_index]
        _nominal = False
        for _column_value in pd.unique(_column_values):
            if not is_number(_column_value):
                _nominal = True
                break
        if _nominal:
            _x[_column_index] = str(_column_index) + '_' + _x[_column_index].astype(str)
        else:
            _x[_column_index] = _x[_column_index].astype(float)

    return _x, _y


def is_number(_i):
    try:
        float(_i)
        return True
    except ValueError:
        return False


class Node:
    def __init__(self, _x, _y, _depth=0):
        self.depth = _depth

        _x_dummies = pd.get_dummies(_x, prefix='dummy')
        self.x_columns = _x_dummies.columns

        # compute current mse
        self.linear_model = LinearRegression()
        self.linear_model.fit(_x_dummies, _y)
        self.mse = mean_squared_error(_y, self.linear_model.predict(_x_dummies))

        self.criterion_mse = float('inf')
        self.criterion_column_index = None
        self.criterion_column_value = None
        self.criterion_numeric = None
        self.children = None

        # recursive tree creator
        if MIN_SAMPLES_TO_SPLIT <= _x.shape[0] and _depth <= MAX_DEPTH:
            self.find_best_split(_x, _y)

    def find_best_split(self, _x, _y):
        # for each column
        for _column_index in \
                tqdm(range(_x.shape[1]), desc='Depth: ' + str(self.depth) + ', # of rows: ' + str(_x.shape[0])):
            _column_values = _x[_column_index]

            # check if numeric
            if is_number(list(_column_values)[0]):
                # for each value in column
                for _column_value in pd.unique(_column_values):
                    _x_lower_equal, _y_lower_equal = \
                        _x[_column_values <= _column_value], _y[_column_values <= _column_value]
                    _x_greater_than, _y_greater_than = \
                        _x[_column_values > _column_value], _y[_column_values > _column_value]

                    # dead end
                    if _x_lower_equal.empty or _x_greater_than.empty:
                        continue

                    _x_lower_equal_linear_model = LinearRegression()
                    _x_lower_equal_linear_model.fit(pd.get_dummies(_x_lower_equal, prefix='dummy'), _y_lower_equal)
                    _y_predict_lower_equal = \
                        _x_lower_equal_linear_model.predict(pd.get_dummies(_x_lower_equal, prefix='dummy'))
                    _x_lower_equal_mse = mean_squared_error(_y_lower_equal, _y_predict_lower_equal)

                    _x_greater_than_linear_model = LinearRegression()
                    _x_greater_than_linear_model.fit(pd.get_dummies(_x_greater_than, prefix='dummy'), _y_greater_than)
                    _y_predict_greater_than = \
                        _x_greater_than_linear_model.predict(pd.get_dummies(_x_greater_than, prefix='dummy'))
                    _x_greater_than_mse = mean_squared_error(_y_greater_than, _y_predict_greater_than)

                    _normalized_mse = (_x_lower_equal_mse * _x_lower_equal.shape[0] +
                                       _x_greater_than_mse * _x_greater_than.shape[0]) / _x.shape[0]

                    # found better split
                    if _normalized_mse < self.criterion_mse:
                        self.criterion_mse = _normalized_mse
                        self.criterion_column_index = _column_index
                        self.criterion_column_value = _column_value
                        self.criterion_numeric = True

            # not numeric
            else:
                # for each value in column
                _normalized_mse = 0
                for _column_value in pd.unique(_column_values):
                    _x_equal, _y_equal = _x[_column_values == _column_value], _y[_column_values == _column_value]

                    # dead end
                    if _x_equal.empty:
                        continue

                    _x_equal_linear_model = LinearRegression()
                    _x_equal_linear_model.fit(pd.get_dummies(_x_equal, prefix='dummy'), _y_equal)
                    _y_predict_equal = _x_equal_linear_model.predict(pd.get_dummies(_x_equal, prefix='dummy'))
                    _x_equal_mse = mean_squared_error(_y_equal, _y_predict_equal)

                    _normalized_mse += _x_equal_mse * _x_equal.shape[0]

                _normalized_mse = _normalized_mse / _x.shape[0]

                # found better split
                if _normalized_mse < self.criterion_mse:
                    self.criterion_mse = _normalized_mse
                    self.criterion_column_index = _column_index
                    self.criterion_numeric = False

        # make children
        if self.criterion_column_index is not None:
            _column_values = _x[self.criterion_column_index]
            if self.criterion_numeric:
                _x_lower_equal, _y_lower_equal = \
                    _x[_column_values <= self.criterion_column_value], _y[_column_values <= self.criterion_column_value]
                _x_greater_than, _y_greater_than = \
                    _x[_column_values > self.criterion_column_value], _y[_column_values > self.criterion_column_value]

                # stop loop
                if len(_x_lower_equal) == len(_x) or len(_x_greater_than) == len(_x):
                    return

                self.children = [
                    Node(_x_lower_equal, _y_lower_equal, _depth=self.depth + 1),
                    Node(_x_greater_than, _y_greater_than, _depth=self.depth + 1)
                ]
            else:
                self.children = {}
                for _column_value in pd.unique(_column_values):
                    _x_equal, _y_equal = _x[_column_values == _column_value], _y[_column_values == _column_value]

                    # stop loop
                    if len(_x_equal) == len(_x):
                        return

                    self.children[_column_value] = Node(_x_equal, _y_equal, _depth=self.depth + 1)

    def predict(self, _x):
        # batch predictions
        _y_predict = np.empty(_x.shape[0])
        if self.criterion_column_index is not None:
            # move to children
            _column_values = _x[self.criterion_column_index]
            # numeric
            if self.criterion_numeric:
                _x_lower_equal, _x_greater_than = \
                    _x[_column_values <= self.criterion_column_value], _x[_column_values > self.criterion_column_value]
                if len(_x_lower_equal) > 0:
                    _y_predict_lower_equal = self.children[0].predict(_x_lower_equal)
                    _y_predict[_column_values <= self.criterion_column_value] = _y_predict_lower_equal.flatten()
                if len(_x_greater_than) > 0:
                    _y_predict_greater_than = self.children[1].predict(_x_greater_than)
                    _y_predict[_column_values > self.criterion_column_value] = _y_predict_greater_than.flatten()
            # not numeric
            else:
                for _column_value in pd.unique(_column_values):
                    _x_equal = _x[_column_values == _column_value]

                    # value does not exist, prepare x and return current
                    if _column_value not in self.children:
                        _x_equal_new = pd.DataFrame(
                            data=pd.get_dummies(_x_equal, prefix='dummy'),
                            columns=self.x_columns).fillna(0)
                        _y_predict[_column_values == _column_value] = \
                            self.linear_model.predict(pd.get_dummies(_x_equal_new, prefix='dummy'))
                    # value exists, move to children
                    else:
                        _y_predict_equal = self.children[_column_value].predict(_x_equal)
                        _y_predict[_column_values == _column_value] = _y_predict_equal.flatten()
        else:
            # no children, return current
            _x_new = pd.DataFrame(data=pd.get_dummies(_x, prefix='dummy'), columns=self.x_columns).fillna(0)
            return self.linear_model.predict(pd.get_dummies(_x_new, prefix='dummy'))

        return _y_predict


class LinearRegressionTree:
    def __init__(self):
        self.root = None

    # creates tree based on data
    def fit(self, _x, _y):
        self.root = Node(_x, _y)

    def predict(self, _x):
        return self.root.predict(_x)


def main():
    for _data_file in os.listdir('data'):
        print('Data:', _data_file)
        _x, _y = read_and_process_data(os.path.join('data', _data_file))

        # time, train set, test set
        _averages = [[0, 0, 0], [0, 0, 0]]
        _k_fold = KFold(n_splits=K_FOLD_SPLITS)
        for _k_fold_index, (_train_index, _test_index) in enumerate(_k_fold.split(_x)):
            print('K Fold:', _k_fold_index)
            _x_train, _x_test = _x.iloc[_train_index], _x.iloc[_test_index]
            _y_train, _y_test = _y.iloc[_train_index], _y.iloc[_test_index]

            # our model
            _start_time = time.time()
            _linear_regression_tree = LinearRegressionTree()
            _linear_regression_tree.fit(_x_train, _y_train)
            _averages[0][0] += time.time() - _start_time

            _y_train_predict = _linear_regression_tree.predict(_x_train)
            _averages[0][1] += mean_squared_error(_y_train, _y_train_predict)

            _y_test_predict = _linear_regression_tree.predict(_x_test)
            _averages[0][2] += mean_squared_error(_y_test, _y_test_predict)

            # scikit-learn model
            _start_time = time.time()
            _decision_regression_tree = \
                DecisionTreeRegressor(min_samples_split=MIN_SAMPLES_TO_SPLIT, max_depth=MAX_DEPTH)
            _decision_regression_tree.fit(pd.get_dummies(_x_train, prefix='dummy'), _y_train)
            _averages[1][0] += time.time() - _start_time

            _y_train_predict = _decision_regression_tree.predict(pd.get_dummies(_x_train, prefix='dummy'))
            _averages[1][1] += mean_squared_error(_y_train, _y_train_predict)

            _x_train_columns = list(pd.get_dummies(_x_train, prefix='dummy').columns)
            _x_test_new = pd.DataFrame(data=pd.get_dummies(_x_test, prefix='dummy'), columns=_x_train_columns).fillna(0)
            _y_test_predict = _decision_regression_tree.predict(_x_test_new)
            _averages[1][2] += mean_squared_error(_y_test, _y_test_predict)

        print('Our model')
        print('Time:', _averages[0][0] / K_FOLD_SPLITS)
        print('Train set MSE:', _averages[0][1] / K_FOLD_SPLITS)
        print('Test set MSE:', _averages[0][2] / K_FOLD_SPLITS)

        print('scikit-learn model:')
        print('Time:', _averages[1][0] / 10)
        print('Train set MSE:', _averages[1][1] / K_FOLD_SPLITS)
        print('Test set MSE:', _averages[1][2] / K_FOLD_SPLITS)

        print('Train set difference:', (_averages[0][1] - _averages[1][1]) / K_FOLD_SPLITS)
        print('Test set difference:', (_averages[0][2] - _averages[1][2]) / K_FOLD_SPLITS)


if __name__ == '__main__':
    main()
