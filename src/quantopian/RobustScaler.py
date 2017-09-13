from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.sparsefuncs import inplace_column_scale
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted

import numpy as np


def _handle_zeros_in_scale(scale, copy=True):
    """ Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features."""

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


class RobustScaler(BaseEstimator, TransformerMixin):
    """
    Copy-Pasta from scikit-learn, since apparently it doesn't exist in older versions.
    """

    def __init__(self, with_centering=True, with_scaling=True,
                 quantile_range=(25.0, 75.0), copy=True):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy

    def _check_array(self, X, copy):
        """Makes sure centering is not enabled for sparse matrices."""
        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        estimator=self, dtype=FLOAT_DTYPES)

        if sparse.issparse(X):
            if self.with_centering:
                raise ValueError(
                    "Cannot center sparse matrices: use `with_centering=False`"
                    " instead. See docstring for motivation and alternatives.")
        return X

    def fit(self, X, y=None):
        """Compute the median and quantiles to be used for scaling.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the median and quantiles
            used for later scaling along the features axis.
        """
        if sparse.issparse(X):
            raise TypeError("RobustScaler cannot be fitted on sparse inputs")
        X = self._check_array(X, self.copy)
        if self.with_centering:
            self.center_ = np.median(X, axis=0)

        if self.with_scaling:
            q_min, q_max = self.quantile_range
            if not 0 <= q_min <= q_max <= 100:
                raise ValueError("Invalid quantile range: %s" %
                                 str(self.quantile_range))

            q = np.percentile(X, self.quantile_range, axis=0)
            self.scale_ = (q[1] - q[0])
            self.scale_ = _handle_zeros_in_scale(self.scale_, copy=False)
        return self

    def transform(self, X):
        """Center and scale the data.
        Can be called on sparse input, provided that ``RobustScaler`` has been
        fitted to dense input and ``with_centering=False``.
        Parameters
        ----------
        X : {array-like, sparse matrix}
            The data used to scale along the specified axis.
        """
        if self.with_centering:
            check_is_fitted(self, 'center_')
        if self.with_scaling:
            check_is_fitted(self, 'scale_')
        X = self._check_array(X, self.copy)

        if sparse.issparse(X):
            if self.with_scaling:
                inplace_column_scale(X, 1.0 / self.scale_)
        else:
            if self.with_centering:
                X -= self.center_
            if self.with_scaling:
                X /= self.scale_
        return X

    def inverse_transform(self, X):
        """Scale back the data to the original representation
        Parameters
        ----------
        X : array-like
            The data used to scale along the specified axis.
        """
        if self.with_centering:
            check_is_fitted(self, 'center_')
        if self.with_scaling:
            check_is_fitted(self, 'scale_')
        X = self._check_array(X, self.copy)

        if sparse.issparse(X):
            if self.with_scaling:
                inplace_column_scale(X, self.scale_)
        else:
            if self.with_scaling:
                X *= self.scale_
            if self.with_centering:
                X += self.center_
        return X