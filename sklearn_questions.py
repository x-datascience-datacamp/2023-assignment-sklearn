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

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances

from pandas.api.types import is_datetime64_any_dtype
from collections import Counter


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fitting function.

         Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to train the model (X_train).
        y : ndarray, shape (n_samples,)
            Labels associated with the training data (y_train).

        Returns
        ----------
        self : instance of KNearestNeighbors
            The current instance of the classifier
        """
        # Checks

        """
        The check_classification_targets function ensures that target y is of
        a non-regression type. Only the following target types
        (as defined in type_of_target) are allowed: 'binary', 'multiclass',
        'multiclass-multioutput', 'multilabel-indicator',
        'multilabel-sequences'.
        """
        check_classification_targets(y)

        """
        The check_X_y function performs an input validation for standard
        estimators (=models). It checks X and y for consistent length,
        enforces X to be 2D and y 1D. By default, X is checked to be non-empty
        and containing only finite values. Standard input checks are also
        applied to y, such as checking that y does not have np.nan or np.inf
        targets.For multi-label y, set multi_output=True to allow 2D and
        sparse y. If the dtype of X is object, attempt converting to
        float, raising on failure.
        """
        X, y = check_X_y(X, y)

        # Number of instances
        self.n_features_in_ = X.shape[1]
        # Classes of y
        self.classes_ = np.unique(y)
        # Instance of X and y as objects of the class
        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on (X_test).

        Returns
        ----------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample (y_test).
        """

        """
        The check_is_fitted function is a sklearn.utils.validation function
        used to check whether an estimator (such as a classifier or regressor)
        has been fitted, i.e. whether it has been trained on input data.
        If the estimator has not been fitted, check_is_fitted will throw
        an error.
        """
        check_is_fitted(self)

        """
        The check_array function is a sklearn.utils.validation function used
        to validate whether an input array is suitable for use in scikit-learn
        estimators. This function checks several things, such as whether the
        array is numeric, whether it has a specific number of dimensions
        (e.g. 2D for arrays), and whether it contains missing values
        (NaN or infinite), among other checks. If the array does not meet
        these requirements, check_array will throw an error.
        """
        check_array(X)

        # Calculate pairwise distances
        """
        pairwise_distances(X, self.X_):
        This function computes the distance from each sample in X (X_test)
        to every sample in self.X_ (X_train). X is the data for which
        predictions are being made, and self.X_ is the training data that
        the model has been fitted on. The result is a distance matrix
        dist_matrix where each entry [i, j] represents the distance between
        the i-th sample in X and the j-th sample in self.X_.
        """
        dist_matrix = pairwise_distances(X, self.X_)

        # Find Indices of Nearest Neighbors
        """
        np.argsort(dist_matrix, axis=1):
        This function sorts each row of dist_matrix in ascending order and
        returns the indices of the sorted elements. The sorting is done
        row-wise, meaning for each sample in X, we get the indices of the
        training samples (self.X_) in order of increasing distance.

        [:, :self.n_neighbors]: This slicing operation takes the first
        self.n_neighbors indices for each row. These are the indices of
        the nearest neighbors.
        """
        dist_sort_pos = np.argsort(dist_matrix, axis=1)[:, :self.n_neighbors]

        # Find Indices of Nearest Neighbors
        """
        np.argsort(dist_matrix, axis=1):
        This function sorts each row of dist_matrix in ascending order and
        returns the indices of the sorted elements. The sorting is done
        row-wise, meaning for each sample in X, we get the indices of the
        training samples (self.X_) in order of increasing distance.

        [:, :self.n_neighbors]:
        This slicing operation takes the first self.n_neighbors
        indices for each row. These are the indices of the nearest neighbors.
        """

        # Get labels of nearest neighbors
        """
        self.y_ (y_train) is the array of labels corresponding to the training
        data self.X_, and self.y_[dist_sort_pos] uses the indices in
        dist_sort_pos to gather the labels of the nearest neighbors for
        each sample in X.
        """
        y_closest = self.y_[dist_sort_pos]

        # Determine predicted values
        """
        This line predicts the label for each sample in X based on the
        majority vote among its nearest neighbors.

        Counter(row):
        For each row in y_closest, a Counter object is
        created to count the frequency of each label among the nearest
        neighbors.

        max(Counter(row), key=Counter(row).get):
        This finds the label with the highest frequency (the most
        common label) among the nearest neighbors for each sample in X.
        """
        y_pred = [max(Counter(row), key=Counter(row).get) for row in y_closest]

        return np.array(y_pred)

    def score(self, X, y):
        """Calculate the score of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to score on (X_test).
        y : ndarray, shape (n_samples,)
            target values (Y_test).

        Returns
        ----------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """
        check_is_fitted(self)
        check_array(X)
        check_classification_targets(y)
        preds = self.predict(X)

        # pres.shape[0] is the number of instances predicted, like len
        return (preds == y).sum() / preds.shape[0]


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
        X = X.reset_index()

        if not is_datetime64_any_dtype(X[self.time_col]):
            raise ValueError("Not in a datetimeFormat")

        date = X[self.time_col]
        date_y_m = date.apply(lambda x: str(x.year) + str(x.month))

        return date_y_m.unique().shape[0] - 1

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
        X = X.reset_index()
        X = X.sort_values(by=self.time_col)
        n_splits = self.get_n_splits(X, y, groups)
        date = X[self.time_col]
        date_y_m = date.apply(lambda x: str(x.year) + str(x.month)).unique()
        for i in range(n_splits):
            year, month = int(date_y_m[i][0:4]), int(date_y_m[i][4:])
            train_idx = X[
                (date.dt.year == year) & (date.dt.month == month)
                ].index.to_numpy()
            year, month = int(date_y_m[i+1][0:4]), int(date_y_m[i+1][4:])
            test_idx = X[
                (date.dt.year == year) & (date.dt.month == month)
                ].index.to_numpy()

            yield train_idx, test_idx
