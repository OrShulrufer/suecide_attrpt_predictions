from external_libs_imports import *

import numpy as np
# Sklearn Model Selection
from sklearn.model_selection import (
    BaseCrossValidator
)
# Sklearn Utilities
from sklearn.utils.validation import indexable


# Sklearn Preprocessing

# Sklearn Metrics

# Sklearn Calibration

# Sklearn Models

# XGBoost
class CustomCV(BaseCrossValidator):
    def __init__(self, n_splits=5, len_train=None):
        self.n_splits = n_splits
        self.len_train = len_train

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        n_samples = len(X)
        indices = np.arange(n_samples)

        for test_fold in range(self.n_splits):
            train_indices = indices[:self.len_train]
            val_start = self.len_train + test_fold * ((n_samples - self.len_train) // self.n_splits)
            val_end = self.len_train + (test_fold + 1) * ((n_samples - self.len_train) // self.n_splits)
            val_indices = indices[val_start:val_end]
            yield train_indices, val_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
