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

# from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.multiclass import unique_labels

from dateutil.relativedelta import relativedelta


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

        self.X_ = X
        self.y_ = y
        self.dtype_ = y.dtype

        self.n_features_in_ = X.shape[1]
        self.classes_ = unique_labels(y)

        return self

    def distance_func(x1: np.ndarray, x2: np.ndarray) -> np.float64:
        """Compute the euclidian distance.

        Parameters
        -----------
        x1,x2 : one dimensionnal vectors, shape (n_features)
            The two vectors for which we calculate the distance

        Returns
        -----------
        res : the distance between the two vectors.
        """
        return np.linalg.norm(x1 - x2)

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

        y_pred = np.empty(X.shape[0], dtype=self.dtype_)

        for i, X_predict in enumerate(X):
            # the euclidian distance for each sample of X_train
            distances = np.array(
                [
                    KNearestNeighbors.distance_func(X_predict, X_train)
                    for X_train in self.X_
                ]
            )

            # indices for each sample sorted by its distance (the k-nearest)
            indexes_sort = np.argsort(distances)[: self.n_neighbors]

            # the labels of the nearest neighbors
            labels_nn = self.y_[indexes_sort]

            unique, counts = np.unique(labels_nn, return_counts=True)

            index_max_counts = np.argmax(counts)

            # the value with the maximum occurrence
            max_occurence = unique[index_max_counts]

            y_pred[i] = max_occurence

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
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        y_pred = self.predict(X)

        return (y == y_pred).mean()


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
        if not (
            isinstance(X, pd.core.frame.DataFrame)
            or isinstance(X, pd.core.series.Series)
        ):
            raise TypeError(
                f"The type of X ({type(X)}) is not consistent \
                      with a pandas dataframe or series"
            )

        if self.time_col == "index":
            if not (
                isinstance(X.index, pd.core.indexes.datetimes.DatetimeIndex)
            ):
                raise ValueError("datetime")

        else:
            if X[self.time_col].dtype != "<M8[ns]":
                raise ValueError("datetime")
            else:
                X.set_index(self.time_col, inplace=True, drop=False)
                self.time_col = "index"

        date = X.index

        self.date_min_ = date.min()
        self.date_max_ = date.max()

        delta = relativedelta(self.date_max_, self.date_min_)
        num_months = delta.years * 12 + delta.months

        return num_months

    def split(self, X, y=None, groups=None):
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
        if not (
            isinstance(X, pd.core.frame.DataFrame)
            or isinstance(X, pd.core.series.Series)
        ):
            raise TypeError(
                f"The type of X ({type(X)}) is not consistent \
                     with a pandas dataframe or series"
            )

        if self.time_col == "index":
            if not (
                isinstance(X.index, pd.core.indexes.datetimes.DatetimeIndex)
            ):
                raise ValueError("datetime")

        else:
            if X[self.time_col].dtype != "<M8[ns]":
                raise ValueError("datetime")
            else:
                X.set_index(self.time_col, inplace=True, drop=False)
                self.time_col = "index"

        n_splits = self.get_n_splits(X, y, groups)

        date_min = self.date_min_

        for i in range(n_splits):
            train_timestamp = date_min + pd.DateOffset(months=i)
            test_timestamp = train_timestamp + pd.DateOffset(months=1)

            idx_train = np.where(X.index == train_timestamp)[0]
            idx_test = np.where(X.index == test_timestamp)[0]
            yield (idx_train, idx_test)
