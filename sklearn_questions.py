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
from sklearn.metrics import euclidean_distances


# hérite baseestimation et classifmixin = interfaces
class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1): # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the model to the training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data.
        y : ndarray, shape (n_samples,)
            Labels associated with the training data.

        Returns
        -------
        self : instance of KNearestNeighbors
            The fitted classifier.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)  # validation forme des données
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]  # nb de features dans le train
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """Predict class labels for test data.
        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Test data.

        Returns
        -------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """
        # prédit la classe de sortie:
        # pour chaque sample test X je dois:
        # - calculer la distance euclidienne entre le train et X
        # - identifier l'indice du plus proche voisin
        # - récup la classe
        # - compte les prédiction et retourne celle qui revient le plus
        def predict_single(sample):
            # indices des plus proches voisins -> distance euclidienne
            closest_indices = np.argsort(
                euclidean_distances(
                    sample.reshape(1, -1),
                    self.X_train_))[
                              0, :self.n_neighbors]
            # reshape = fait une matrice à 1 ligne :
            # vérifie que les tailles matchent entre train et test
            # récup labels
            nearest_labels = self.y_train_[closest_indices]
            # compte les labels et récupère le plus proche
            unique_labels, label_counts = np.unique(
                nearest_labels, return_counts=True)
            return unique_labels[np.argmax(label_counts)]

        # check des données sinon ça pète
        check_is_fitted(self)
        X = check_array(X)

        # applique le predict sur toutes les lignes de X :
        # retourne un tableau avec une pred pour chaque
        predictions = np.apply_along_axis(predict_single, 1, X)
        return predictions

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        return np.mean(self.predict(X) == y)


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

    # - split en train/test en cv : mois 1 et mois 2 entrainer et tester
    # - nb split = nb_mois-1

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
        if not X[self.time_col].dtype == 'datetime64[ns]':
            raise ValueError('time_col should be of type datetime64[ns]')
        return X[self.time_col].dt.to_period('M').nunique() - 1

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
        X = (X.reset_index()
             .resample('M', on=self.time_col)
             .apply(lambda x: x.index))
        for i in range(n_splits):
            yield X.iloc[i].values, X.iloc[i + 1].values
