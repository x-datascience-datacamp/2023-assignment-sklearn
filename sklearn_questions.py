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
from pandas.api.types import is_datetime64_any_dtype as is_datetime

import scipy.stats

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import (
    check_classification_targets,
)
from sklearn.metrics.pairwise import pairwise_distances, check_pairwise_arrays
from sklearn.metrics import accuracy_score


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
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.n_features_in_ = X.shape[1]
        self.X_ = X
        self.classes_, self.y_ = np.unique(y, return_inverse=True)

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
        check_is_fitted(self)
        X = check_array(X)
        check_pairwise_arrays(X, self.X_)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Shape of input is different from what was seen in" " `fit`"
            )

        dist = pairwise_distances(X, self.X_)
        idx = np.argsort(dist, axis=1)[:, : self.n_neighbors]
        mode, _ = scipy.stats.mode(self.y_[idx], axis=1)
        y_pred = self.classes_[mode.ravel().astype(int)]
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
        check_is_fitted(self)
        check_classification_targets(y)
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Shape of input is different from what was seen in" " `fit`"
            )
        return accuracy_score(y, self.predict(X))


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

    def __init__(self, time_col="index"):  # noqa: D107
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
        _X = pd.DataFrame(X)
        if "index" in _X.columns:
            raise ValueError(
                "Column 'index' already exists. Please choose a different"
                " time_col or rename the existing 'index' column."
            )
        _X = _X.reset_index(names="index")
        if not is_datetime(_X[self.time_col]):
            raise ValueError(
                f"Column {self.time_col} is not a datetime column."
            )
        max_date = _X[self.time_col].max()
        min_date = _X[self.time_col].min()
        return (
            12 * (max_date.year - min_date.year)
            + max_date.month
            - min_date.month
        )

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
        n_splits = self.get_n_splits(X, y, groups)
        _X = pd.DataFrame(X)
        if "index" in _X.columns:
            raise ValueError(
                "Column 'index' already exists. Please choose a different"
                " time_col or rename the existing 'index' column."
            )
        _X = _X.reset_index(names="index")
        if not is_datetime(_X[self.time_col]):
            raise ValueError(
                f"Column {self.time_col} is not a datetime column."
            )
        first_date = _X[self.time_col].min()
        first_month, first_year = (
            first_date.month - 1,  # index months from 0
            first_date.year,
        )
        for i in range(n_splits):
            train_month = (first_month + i) % 12 + 1
            train_year = first_year + (first_month + i) // 12
            idx_train = _X.loc[
                (_X[self.time_col].dt.month == train_month)
                & (_X[self.time_col].dt.year == train_year)
            ].index.values
            idx_test = _X.loc[
                (_X[self.time_col].dt.month == (train_month % 12 + 1))
                & (_X[self.time_col].dt.year == train_year + train_month // 12)
            ].index.values

            yield (idx_train, idx_test)
