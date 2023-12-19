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
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.multiclass import unique_labels


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
        check_classification_targets(y)
        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        ----------
        y_pred : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """
        check_is_fitted(self)

        X = check_array(X)

        dist = pairwise_distances(X, self.X_, metric='euclidean', n_jobs=-1)
        neighbor_idx = self.y_[np.argsort(dist)[:, :self.n_neighbors]]
        y_pred = np.apply_along_axis(self._most_common_integer,
                                     axis=1, arr=neighbor_idx)
        return np.array(y_pred, dtype=self.classes_.dtype)

    def _most_common_integer(self, arr):
        """Calculate the most frequent integer in an array.

        Will be used to compute the most frequent neighbor of a sample.

        Parameters
        ----------
        arr : ndarray, shape (n_neighbors,)
            Classes of neighbors.

        Returns
        ----------
        most_frequent : object
            Most frequent element in the array;
            has the same class as the elements of the input array.
        """
        unique_elements, counts = np.unique(arr, return_counts=True)
        most_frequent_index = np.argmax(counts)
        most_frequent = unique_elements[most_frequent_index]
        return most_frequent

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
        check_classification_targets(y)
        X, y = check_X_y(X, y)

        pred = self.predict(X)
        score = np.mean(pred == y)
        return score


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
        if self.time_col != 'index':
            self.X_ = X.reset_index()
            self.X_ = self.X_.set_index(self.time_col)
        else:
            self.X_ = X.copy()

        if not is_datetime(self.X_.index):
            raise ValueError("The reference column must be of type 'datetime'")

        unique_months = self.X_.index.to_period("M").nunique()
        n_splits = unique_months - 1
        return n_splits

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

        period = self.X_.index.to_period('M')
        unique_months = period.unique().sort_values()
        indices_of_index = np.array(range(len(self.X_)))

        cur_index = 0
        cur_month = unique_months[cur_index]
        idx_test = None
        for i in range(n_splits):
            if i == 0:
                idx_train = self._find_index_target_month(indices_of_index,
                                                          period, cur_month)
            else:
                idx_train = idx_test

            cur_index += 1
            cur_month = unique_months[cur_index]

            idx_test = self._find_index_target_month(indices_of_index,
                                                     period, cur_month)
            yield (
                idx_train, idx_test
            )

    def _find_index_target_month(self, indices_of_index, period, target_month):
        """Find the indices of original matrix corresponding to target month.

        The indices returned correpond to elements, such that their value in
        self.time_col corresponds to target_month.

        Parameters
        ----------
        indices_of_index : ndarray, shape (X.shape[0],)
            Range from 0 to X.shape[0]
        period: pandas.core.indexes.period.PeriodIndex, shape (X.shape[0],)
            Dates in self.X_.index in format 'YYYY-MM'
        target_month: pandas._libs.tslibs.period.Period
            Month target, looks like Period('YYYY-MM', 'M')

        Returns
        ----------
        indices_target_month : ndarray, shape (n,)
            Indices of original matrix with a
            date corresponding to target_month.
            Can be empty.
        """
        mask = period == target_month
        indices_target_month = indices_of_index[mask]
        return indices_target_month
