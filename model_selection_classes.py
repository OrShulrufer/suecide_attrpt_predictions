from external_libs_imports import *
from model_selection_functions import wald_test

# Custom transformer for Wald Test-based feature selection
class WaldTestFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, model, alpha=0.05):
        self.model = model
        self.alpha = alpha

    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def transform(self, X):
        p_values = wald_test(X, self.model)
        selected_features = p_values < self.alpha
        return X[:, selected_features]


# Define CustomRFClassifier
class CustomRFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier=None, threshold=0.5, **kwargs):
        self.classifier = classifier if classifier is not None else RandomForestClassifier(**kwargs)
        self.threshold = threshold

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        proba = self.classifier.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def set_params(self, **params):
        if 'classifier' in params:
            self.classifier = params.pop('classifier')
        if 'threshold' in params:
            self.threshold = params.pop('threshold')
        self.classifier.set_params(**params)
        return self

    def get_params(self, deep=True):
        params = self.classifier.get_params(deep)
        params['threshold'] = self.threshold
        params['classifier'] = self.classifier
        return params

