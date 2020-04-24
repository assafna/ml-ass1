import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

MIN_OBS_TO_SPLIT = 10


def read_and_process_data(_file):
    _data = pd.read_csv(_file, header=None)

    # clean
    _data = _data.replace('?', np.nan)
    _data = _data.dropna()

    # replace nominal with new columns
    _x, _y = _data.iloc[:, :-1], _data.iloc[:, -1]

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

        # compute current mse
        self.linear_model = LinearRegression()
        self.linear_model.fit(pd.get_dummies(_x), _y)
        self.mse = mean_squared_error(_y, self.linear_model.predict(pd.get_dummies(_x)))

        self.criterion_mse = float('inf')
        self.criterion_column_index = None
        self.criterion_column_value = None
        self.criterion_numeric = None
        self.children = None

        # recursive tree creator
        if MIN_OBS_TO_SPLIT <= _x.shape[0]:
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
                    _x_lower_equal_linear_model.fit(pd.get_dummies(_x_lower_equal), _y_lower_equal)
                    _y_predict_lower_equal = \
                        _x_lower_equal_linear_model.predict(pd.get_dummies(_x_lower_equal))
                    _x_lower_equal_mse = mean_squared_error(_y_lower_equal, _y_predict_lower_equal)

                    _x_greater_than_linear_model = LinearRegression()
                    _x_greater_than_linear_model.fit(pd.get_dummies(_x_greater_than), _y_greater_than)
                    _y_predict_greater_than = \
                        _x_greater_than_linear_model.predict(pd.get_dummies(_x_greater_than))
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
                    _x_equal_linear_model.fit(pd.get_dummies(_x_equal), _y_equal)
                    _y_predict_equal = _x_equal_linear_model.predict(pd.get_dummies(_x_equal))
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
                self.children = [
                    Node(_x_lower_equal, _y_lower_equal, _depth=self.depth + 1),
                    Node(_x_greater_than, _y_greater_than, _depth=self.depth + 1)
                ]
            else:
                self.children = {}
                for _column_value in pd.unique(_column_values):
                    _x_equal, _y_equal = _x[_column_values == _column_value], _y[_column_values == _column_value]
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
                _y_predict_lower_equal, _y_predict_greater_than = \
                    self.children[0].predict(_x_lower_equal), self.children[1].predict(_x_greater_than)
                _y_predict[_column_values <= self.criterion_column_value] = _y_predict_lower_equal.flatten()
                _y_predict[_column_values > self.criterion_column_value] = _y_predict_greater_than.flatten()
            # not numeric
            else:
                for _column_value in pd.unique(_column_values):
                    _x_equal = _x[_column_values == _column_value]
                    _y_predict_equal = self.children[_column_value].predict(_x_equal)
                    _y_predict[_column_values == _column_value] = _y_predict_equal.flatten()
        else:
            # no children, return current
            return self.linear_model.predict(pd.get_dummies(_x))

        return _y_predict


class DecisionRegressionTree:
    def __init__(self):
        self.root = None

    # creates tree based on data
    def fit(self, _x, _y):
        self.root = Node(_x, _y)

    def predict(self, _x):
        return self.root.predict(_x)


def main():
    _start_time = time.time()
    _x, _y = read_and_process_data('imports-85.data')
    _x_train, _x_test, _y_train, _y_test = train_test_split(_x, _y, test_size=0.2)
    _decision_regression_tree = DecisionRegressionTree()
    _decision_regression_tree.fit(_x_train, _y_train)
    print('Training time:', time.time() - _start_time)

    _y_train_predict = _decision_regression_tree.predict(_x_train)
    print('Train set MSE:', mean_squared_error(_y_train, _y_train_predict))
    _y_test_predict = _decision_regression_tree.predict(_x_test)
    print('Test set MSE:', mean_squared_error(_y_test, _y_test_predict))


if __name__ == '__main__':
    main()
