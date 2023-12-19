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
        X, y = check_X_y(X, y)
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
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """
        #y_pred = np.zeros(X.shape[0])
        #return y_pred

        check_is_fitted(self, ['X_', 'y_'])
        
        # Input validation
        X = check_array(X)

        # Compute distances between X and training data
        distances = pairwise_distances(X, self.X_, metric='euclidean')

        # Find the indices of the nearest neighbors
        nearest_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]

        # Predict the class of each sample
        y_pred = np.array([np.bincount(self.y_[indices]).argmax() for indices in nearest_indices])
        
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
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X, y = check_X_y(X, y)

        # Predict the class labels
        y_pred = self.predict(X)

        # Calculate accuracy
        return np.mean(y_pred == y)

from sklearn.model_selection import BaseCrossValidator
import pandas as pd
import numpy as np

class MonthlySplit(BaseCrossValidator):
    def __init__(self, time_col='index'):
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        if self.time_col == 'index':
            if not pd.api.types.is_datetime64_any_dtype(X.index):
                raise ValueError("Index is not a datetime type")
            dates = X.index
        else:
            if not pd.api.types.is_datetime64_any_dtype(X[self.time_col]):
                raise ValueError(f"{self.time_col} is not a datetime type")
            dates = X[self.time_col]

        unique_months = len(np.unique(dates.to_period('M')))
        return max(0, unique_months - 1)

    def split(self, X, y=None, groups=None):
        if self.time_col == 'index':
            if not pd.api.types.is_datetime64_any_dtype(X.index):
                raise ValueError("Index is not a datetime type")
            dates = X.index
        else:
            if not pd.api.types.is_datetime64_any_dtype(X[self.time_col]):
                raise ValueError(f"{self.time_col} is not a datetime type")
            dates = X[self.time_col]

        months = dates.to_period('M')
        unique_months = sorted(np.unique(months))

        for i in range(len(unique_months) - 1):
            train_mask = months == unique_months[i]
            test_mask = months == unique_months[i + 1]

            idx_train = np.where(train_mask)[0]
            idx_test = np.where(test_mask)[0]
            yield (idx_train, idx_test)
