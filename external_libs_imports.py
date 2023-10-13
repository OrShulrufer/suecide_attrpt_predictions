import joblib
import seaborn as sns
import re
import warnings
from scipy.stats import shapiro, randint as sp_randint, chi2, ttest_ind, ks_2samp
from scipy.spatial import distance
import numpy as np
import pandas as pd

# Sklearn Utilities
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.utils import resample, shuffle
from sklearn.utils.validation import indexable
from sklearn.utils.multiclass import type_of_target
from sklearn.base import clone, BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.inspection import permutation_importance

# Sklearn Model Selection
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV,
    cross_val_score, train_test_split,
    StratifiedKFold, PredefinedSplit, KFold,
    GroupShuffleSplit, BaseCrossValidator, BaseShuffleSplit
)

# Sklearn Preprocessing
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, LabelEncoder,
    OneHotEncoder, MinMaxScaler, QuantileTransformer,
    FunctionTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sklearn Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    balanced_accuracy_score, cohen_kappa_score,
    roc_curve, confusion_matrix, brier_score_loss,
    make_scorer, log_loss, auc, precision_recall_curve, classification_report
)

# Sklearn Calibration
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# Sklearn Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, HistGradientBoostingClassifier, VotingClassifier
)
from sklearn.svm import SVC, LinearSVC

# XGBoost
from xgboost import XGBClassifier

