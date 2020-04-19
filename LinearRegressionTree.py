import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

MIN_OBS_TO_SPLIT = 5


def read_and_process_data(_file):
    _data = pd.read_csv(_file, header=None)

    # clean
    _data = _data.replace('?', np.nan)
    _data = _data.dropna()

    # replace nominal with new columns
    _x, _y = pd.get_dummies(_data.iloc[:, :-1]), _data.iloc[:, -1]
    _x, _y = pd.DataFrame(_x.values), pd.DataFrame(_y.values)

    return _x, _y


class Node:
    def __init__(self, _x, _y):
        # compute current mse
        self.linear_model = LinearRegression()
        self.linear_model.fit(_x, _y)
        self.mse = mean_squared_error(_y, self.linear_model.predict(_x))

        self.criterion_mse = float('inf')
        self.criterion_column_index = None
        self.criterion_column_value = None
        self.children = None

        # recursive tree creator
        if MIN_OBS_TO_SPLIT <= _x.shape[0]:
            self.find_best_split(_x, _y)

    def find_best_split(self, _x, _y):
        # for each column
        for _column_index in range(_x.shape[1]):
            _column_values = _x[_column_index]
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
                _x_lower_equal_linear_model.fit(_x_lower_equal, _y_lower_equal)
                _y_predict_lower_equal = _x_lower_equal_linear_model.predict(_x_lower_equal)
                _x_lower_equal_mse = mean_squared_error(_y_lower_equal, _y_predict_lower_equal)

                _x_greater_than_linear_model = LinearRegression()
                _x_greater_than_linear_model.fit(_x_greater_than, _y_greater_than)
                _y_predict_greater_than = _x_greater_than_linear_model.predict(_x_greater_than)
                _x_greater_than_mse = mean_squared_error(_y_greater_than, _y_predict_greater_than)

                _normalized_mse = (_x_lower_equal_mse * _x_lower_equal.shape[0] +
                                   _x_greater_than_mse * _x_greater_than.shape[0]) / _x.shape[0]

                # found better split
                if _normalized_mse < self.criterion_mse:
                    self.criterion_mse = _normalized_mse
                    self.criterion_column_index = _column_index
                    self.criterion_column_value = _column_value

        # make children
        if self.criterion_column_index is not None:
            _column_values = _x[self.criterion_column_index]
            _x_lower_equal, _y_lower_equal = \
                _x[_column_values <= self.criterion_column_value], _y[_column_values <= self.criterion_column_value]
            _x_greater_than, _y_greater_than = \
                _x[_column_values > self.criterion_column_value], _y[_column_values > self.criterion_column_value]
            self.children = [
                Node(_x_lower_equal, _y_lower_equal),
                Node(_x_greater_than, _y_greater_than)
            ]

    def predict(self, _x):
        # batch predictions
        _y_predict = np.empty(_x.shape[0])
        if self.criterion_column_index is not None:
            # move to children
            _column_values = _x[self.criterion_column_index]
            _x_lower_equal, _x_greater_than = \
                _x[_column_values <= self.criterion_column_value], _x[_column_values > self.criterion_column_value]
            _y_predict_lower_equal, _y_predict_greater_than = \
                self.children[0].predict(_x_lower_equal), self.children[1].predict(_x_greater_than)
            _y_predict[_column_values <= self.criterion_column_value] = _y_predict_lower_equal.flatten()
            _y_predict[_column_values > self.criterion_column_value] = _y_predict_greater_than.flatten()
        else:
            # no children, return current
            return self.linear_model.predict(_x)

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
    _x, _y = read_and_process_data('machine.data')
    _decision_regression_tree = DecisionRegressionTree()
    _decision_regression_tree.fit(_x, _y)
    print('Training time:', time.time() - _start_time)
    _y_predict = _decision_regression_tree.predict(_x)
    print('MSE:', mean_squared_error(_y, _y_predict), 'Total time:', time.time() - _start_time)


if __name__ == '__main__':
    main()
