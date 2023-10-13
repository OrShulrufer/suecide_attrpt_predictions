from data_preeprocess_functions import scale_columns
from external_libs_imports import *
from model_selection_classes import WaldTestFeatureSelector


model_params = {
    'LogisticRegression': {
        'classifier': LogisticRegression(),
        'params': {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'class_weight': ['balanced'],
            'fit_intercept': [True, False]

        },
        'steps': {
            'scaler': [FunctionTransformer(func=np.vectorize(np.log1p), validate=True), FunctionTransformer(func=scale_columns, validate=True)],
            'wald_feature': WaldTestFeatureSelector(model=LogisticRegression())
        },
        'p_value_support': True,
        'hypothesis_testing': True
    },
    'RandomForestClassifier': {
        'classifier': RandomForestClassifier(),
        'params': {
            'n_estimators': [10, 50, 100, 300],
            'max_depth': [None, 6, 10, 20],
            'criterion': ['gini', 'entropy'],
            'class_weight': ['balanced', 'balanced_subsample'],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
        'steps': {
            'wald_feature': WaldTestFeatureSelector(model=LogisticRegression())
        },
        'p_value_support': False,
        'hypothesis_testing': True
    },
    'DecisionTreeWithFeatureImportance': {
        'classifier': DecisionTreeClassifier(),
        'params': {
            'max_depth': [None, 6, 8, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced'],
            'splitter': ['best', 'random'],
            'criterion': ['gini', 'entropy']
        },
        'steps': {'wald_feature': WaldTestFeatureSelector(model=LogisticRegression())},
        'p_value_support': False,
        'hypothesis_testing': True
    },
    # 'LinearDiscriminantAnalysis': {
    #     'classifier': LinearDiscriminantAnalysis(),
    #     'params': {
    #         'solver': ['svd', 'lsqr', 'eigen'],
    #         'shrinkage': [None, 'auto', 'scale'],
    #         'priors': [None, 'uniform'],
    #         'n_components': [None, 1, 2, 3],
    #         'store_covariance': [True, False]
    #     },
    #     'steps': {},
    #     'p_value_support': False,
    #     'hypothesis_testing': True
    # },
}
