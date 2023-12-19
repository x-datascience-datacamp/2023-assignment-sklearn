"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:

- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `check_*` functions imported in the file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.


Detailed instructions for question 2:
The data to split should contain the index or one column in
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 4 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict on january etc.

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.

Hints
-----
- You can use the function:

from sklearn.metrics.pairwise import pairwise_distances

to compute distances between 2 sets of samples.
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fitting function.

         Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to train the model.
        y : ndarray, shape (n_samples,)
            Labels associated with the training data.

        Returns
        ----------
        self : instance of KNearestNeighbors
            The current instance of the classifier
        """
        check_array(X)
        check_classification_targets(y)
        X, y = check_X_y(X, y)
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.sort(list(set(np.array(y))))
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        ----------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """
        X = check_array(X)
        check_is_fitted(self)
        N = X.shape[0]
        K = self.n_neighbors
        y_pred = np.zeros(N)
        D = pairwise_distances(X, self.X_train_, metric='l2')
        D_sort = np.argsort(D, axis=1)
        D_sort = D_sort[:, 0:K]
        D_final = np.zeros((N, K))
        # print(self.y_train_)
        convert = False
        neg = False
        # print('y_train = {}'.format(self.y_train_))
        if isinstance(self.y_train_[0], type('one')):
            convert = True
            a, b = np.unique(self.y_train_, return_inverse=True)
            # print('classes = {}'.format(self.classes_))
        else:
            if np.min(self.y_train_) < 0:
                neg = True
                b = self.y_train_ + np.abs(np.min(self.y_train_))
            else:
                b = self.y_train_.copy()
        # print('b = {}'.format(b))
        # print('has_attribute_class = {}'.format(hasattr(self, "classes_")))
        for i in range(D_sort.shape[0]):
            for j in range(D_sort.shape[1]):
                # print(D_sort[i, j])
                D_final[i, j] = b[D_sort[i, j]]
        # print(D_final)
        for t in range(N):
            y_pred[t] = np.argmax(np.bincount(D_final[t, :].astype('int')))
        if convert:
            y_pred = np.array([a[int(val)] for val in y_pred])
        if neg:
            y_pred = y_pred - np.abs(np.min(self.y_train_))
        # print(y_pred)
        # print(self.__dict__ == self.__dict__)
        return y_pred

    def score(self, X, y):
        """Calculate the score of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to score on.
        y : ndarray, shape (n_samples,)
            target values.

        Returns
        ----------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """
        check_array(X)
        check_classification_targets(y)
        check_is_fitted(self)
        y_pred = self.predict(X)
        results = np.equal(y_pred, y)
        return np.mean(results)


class MonthlySplit(BaseCrossValidator):
    """CrossValidator based on monthly split.

    Split data based on the given `time_col` (or default to index). Each split
    corresponds to one month of data for the training and the next month of
    data for the test.

    Parameters
    ----------
    time_col : str, defaults to 'index'
        Column of the input DataFrame that will be used to split the data. This
        column should be of type datetime. If split is called with a DataFrame
        for which this column is not a datetime, it will raise a ValueError.
        To use the index as column just set `time_col` to `'index'`.
    """

    def __init__(self, time_col='index'):  # noqa: D107
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        if not isinstance(X, type(pd.DataFrame())):
            # print('X index type = {}'.format(type(X.index)))
            Xn = pd.DataFrame({'date': X.index, 'val': X.values})
            # print(Xn.index[-1])
            Xn['date'] = pd.to_datetime(Xn['date'])
            # print('Xn index type = {}'.format(type(Xn.index)))
        elif self.time_col == 'index' and 'date' not in X.columns[0]:
            Xn = X.reset_index().copy()
            Xn = Xn.rename(columns={'index': 'date'}, inplace=False)
            # print(Xn.columns)
        else:
            Xn = X.copy()
            if 'date' not in Xn.columns[0]:
                print('tutu')
                Xn = Xn.rename({self.time_col: 'date'})
                print('fin tutu')
        month_of_dates = pd.to_datetime(Xn['date']).dt.strftime('%b-%Y')
        # print('result n split = {}'.format(len(set(month_of_dates)) - 1))
        return len(set(month_of_dates)) - 1

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        idx_train : ndarray
            The training set indices for that split.
        idx_test : ndarray
            The testing set indices for that split.
        """
        if self.time_col != 'index':
            if not isinstance(X[self.time_col][0], type(pd.Timestamp('now'))):
                raise ValueError('datetime')
        else:
            if not isinstance(X.index[0], type(pd.Timestamp('now'))):
                raise ValueError('datetime')
        if not isinstance(X, type(pd.DataFrame())):
            # print('X index type = {}'.format(type(X.index)))
            Xn = pd.DataFrame({'date': X.index, 'val': X.values})
            # print(Xn.index[-1])
            Xn['date'] = pd.to_datetime(Xn['date'])
            # print('Xn index type = {}'.format(type(Xn.index)))
        elif self.time_col == 'index':
            Xn = X.reset_index().copy()
            Xn = Xn.rename(columns={'index': 'date'})
        else:
            Xn = X.copy()
            # print(Xn.head())
            if 'date' not in Xn.columns[0]:
                # print('tutu')
                Xn = Xn.rename(columns={self.time_col: 'date'}, inplace=False)
        n_splits = self.get_n_splits(Xn, y, groups)
        Xn['month_year'] = pd.to_datetime(Xn['date']).dt.strftime('%b-%Y')
        months_years = np.unique(np.sort(pd.to_datetime(Xn['month_year'])))
        # print('months_years = {}'.format(months_years))
        Xn['month_year'] = pd.to_datetime(Xn['month_year'])
        Xn = Xn.reset_index()
        for i in range(n_splits):
            idx_train = list(Xn[Xn['month_year'] == months_years[i]].index)
            idx_test = list(Xn[Xn['month_year'] == months_years[i+1]].index)
            # print('train = {}'.format(idx_train))
            # print('test = {}'.format(idx_test))
            yield (
                idx_train, idx_test
            )
