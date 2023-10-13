from sklearn.metrics import mean_squared_error

from external_libs_imports import *

def p_val(random_search):
    # Assuming `clf` is your logistic regression model after fitting
    clf = random_search.best_estimator_.named_steps['classifier']
    p_value = chi2.sf(np.square(clf.coef_) / np.square(clf.intercept_), 1)
    return p_value


# Define a custom scoring function for medical applications with FN minimization, probability quality, and distribution needs
def custom_medical_score(y_true, y_pred, prob_predictions):
    # Calculate precision and recall
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    # Calculate the mean squared error (MSE) of predicted probabilities and the true binary labels
    mse = ((prob_predictions - y_true) ** 2).mean()
    # Adjust weights based on priorities
    # Higher weight for recall to minimize false negatives
    # Higher weight for probability quality (lower MSE) and distribution needs
    alpha = 0.7  # Weight for recall (FN minimization)
    beta = 0.3  # Weight for probability quality and distribution needs
    # Calculate the F1 score with weights
    f1_combined = (1 + alpha ** 2) * (precision * recall) / ((alpha ** 2 * precision) + recall)
    # Apply reliability penalty based on MSE (lower MSE is better)
    combined_score = (1 - mse) * f1_combined

    return combined_score



def custom_scorer(y_true, y_pred, y_prob, debug=False):
    final_score = 0
    debug_info = {}
    # score calculation waits
    alpha_ks = 2.0
    gamma_mse = 0.1
    alpha_precision = 0.1

    # Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)
    debug_info['MSE'] = mse

    # Precision and Recall
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    debug_info['Precision'] = precision
    debug_info['Recall'] = recall

    # Adjusted F1 score
    alpha_precision = 1  # Adjust as needed
    f1_score = (1 + alpha_precision ** 2) * (precision * recall) / (alpha_precision ** 2 * precision + recall)
    debug_info['F1 Score'] = f1_score

    # Combine F1 and MSE
    score = f1_score - gamma_mse * mse
    debug_info['Initial Score'] = score
    final_score += score

    # Conditional Incorporation of Statistical Tests
    if y_prob is not None and np.unique(y_prob).size > 1:
        # Variance of probabilities for equal distribution
        variance = np.var(y_prob)
        debug_info['Variance'] = variance
        final_score += variance

        # P-value (Two-sample t-test)
        _, p_value = ttest_ind(y_true, y_prob)
        if not np.isnan(p_value):
            p_value_score = -np.log(p_value)
            debug_info['P-value Score'] = p_value_score
            final_score += p_value_score

        # KS Test
        ks_statistic, ks_pvalue = ks_2samp(y_true, y_prob)
        if ks_pvalue < 0.05:
            penalty = -alpha_ks * np.log(ks_pvalue)
            final_score += penalty
            debug_info['Hypothesis'] = f'H1 Accepted (KS Test), penalty to score is {penalty}'
        else:
            debug_info['Hypothesis'] = 'H0 Accepted (KS Test), so no penalty to score'
            # Here, you could potentially reward the final_score if needed

    # Print debug information if debug mode is on
    if debug:
        print("Debug Information:", debug_info)

    return final_score


def model_checks(clf_name, clf_in, X_train, y_train, X_check, y_check):
    clf = clf_in
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_check)
    if clf_name == 'HistGradientBoosting' or clf_name == 'LinearSVC':
        # Compute permutation importance
        result = permutation_importance(clf, X_check, y_check, n_repeats=30, random_state=42)
        # Create a feature_importances_-like array
        feature_importances_like = result.importances_mean
        # Sort features by importance
        feature_importence_series = feature_importances_like.argsort()[::-1]
    else:
        result = permutation_importance(clf, X_check, y_check, n_repeats=30, random_state=42)
        # Create a feature_importances_-like array
        feature_importances_like = result.importances_mean
        # Sort features by importance
        feature_importence_series = feature_importances_like.argsort()[::-1]

    perm_importance = permutation_importance(clf, X_check, y_check, n_repeats=30, random_state=42)
    f1 = f1_score(y_check, y_pred)
    recall = recall_score(y_check, y_pred)
    precision = precision_score(y_check, y_pred)
    accuracy = accuracy_score(y_check, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_check, y_pred)
    y_pred_proba = clf.predict_proba(X_check)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_check, y_pred_proba)
    auc_score = roc_auc_score(y_check, y_pred_proba)
    brier = brier_score_loss(y_check, y_pred_proba)


    clf_cal = CalibratedClassifierCV(base_estimator=clf, cv=3, method="sigmoid")
    clf_cal.fit(X_train, y_train)

    precision_t, recall_t, thresholds = precision_recall_curve(y_check, y_pred_proba)
    optimal_t = thresholds[np.argmax(precision_t * recall_t)]
    y_pred_proba_cal = clf_cal.predict_proba(X_check)[::,1]
    y_pred_cal = [1 if prob >= optimal_t else 0 for prob in y_pred_proba_cal]
    f1_opt_t = f1_score(y_check, y_pred_cal)
    recall_opt_t = recall_score(y_check, y_pred_cal)
    precision_opt_t = precision_score(y_check, y_pred_cal)
    accuracy_opt_t = accuracy_score(y_check, y_pred_cal)
    balanced_accuracy_opt_t = balanced_accuracy_score(y_check, y_pred_cal)

    joblib.dump(clf, f"./{clf_name}.joblib")

    rv = {
        'feature_importance': feature_importence_series,
        'f1_score': f1,
        'recall': recall,
        'precision': precision,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'auc_score': auc_score,
        'brier': brier,
        'fpr': fpr,
        'tpr': tpr,
        'f1_opt_t': f1_opt_t,
        'recall_opt_t': recall_opt_t,
        'precision_opt_t': precision_opt_t,
        'accuracy_opt_t': accuracy_opt_t,
        'balanced_accuracy_opt_t': balanced_accuracy_opt_t,
        'y_pred_proba': y_pred_proba,
        'y_pred_proba_cal': y_pred_proba_cal,
        'y_test': y_check,
        'X_test': X_check,
        'perm_importance': perm_importance
    }
    return rv


def investigate_feature_importances(clf, clf_name, X_train):
    # Investigate feature importances
    feature_importances = clf.feature_importances_.sort_values(inplace=True, ascending=False)
    feature_importance_series = pd.Series(feature_importances, index=X_train.columns)
    feature_importance_series.sort_values(inplace=True, ascending=False)
    return feature_importance_series


# Function to perform Wald Test and return p-values
def wald_test(X, model):
    coef = model.coef_[0]
    pred_probs = model.predict_proba(X)[:, 1]
    J = len(coef)  # number of features
    V = np.diag(pred_probs * (1 - pred_probs))
    cov = np.linalg.inv(np.dot(np.dot(X.T, V), X))
    standard_errors = np.sqrt(np.diag(cov))
    wald_statistics = (coef / standard_errors) ** 2
    p_values = chi2.sf(wald_statistics, 1)
    return p_values