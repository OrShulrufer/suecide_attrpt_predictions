# suecide_attrpt_predictions

# Table of Contents

1. **[Introduction](#introduction)**
    - 1.1 *[Background and Motivation](#background-and-motivation)*
    - 1.2 *[Objectives](#objectives)*
    - 1.3 *[Significance of the Project](#significance-of-the-project)*
2. **[Literature Review](#literature-review)**
    - 2.1 *[Supervised Learning](#supervised-learning)*
    - 2.2 *[Zero-Inflated Models](#zero-inflated-models)*
    - 2.3 *[Potability Prediction in Pandemic Scenarios](#potability-prediction-in-pandemic-scenarios)*
3. **[Methodology](#methodology)**
    - 3.1 *[Data Acquisition and Description](#data-acquisition-and-description)*
        - 3.1.1 [Dataset Features](#dataset-features)
            - [Comprehensive Understanding of Dataset Features](#comprehensive-understanding-of-dataset-features)
                - [Data Structure Tables](#data-structure-tables)
                    - [Basic Information](#basic-information)
                    - [Time-Based Features (Multiple Prefixes)](#time-based-features-multiple-prefixes)
                    - [Medical Diagnoses (avh prefix)](#medical-diagnoses-avh-prefix)
                    - [Types of Prescriptions (mrs prefix)](#types-of-prescriptions-mrs-prefix)
        - 3.1.2 [Zero-Inflation in Dataset Features](#zero-inflation-in-dataset-features)
    - 3.2 *[Data Pre-processing](#data-pre-processing)*
        - 3.2.1 [Handling True Zeros](#handling-true-zeros)
        - 3.2.2 [Addressing Non-Normal Distribution](#addressing-non-normal-distribution)
    - 3.3 *[Model Development](#model-development)*
        - 3.3.1 [Pipeline for Pre-Hospitalization Period](#pipeline-for-pre-hospitalization-period)
        - 3.3.2 [Pipeline for Post-Hospitalization Period](#pipeline-for-post-hospitalization-period)
        - 3.3.3 [Incorporation of P-value, Wald Test, and Variance Logic](#incorporation-of-p-value-wald-test-and-variance-logic)
    - 3.4 *[Model Evaluation and Optimization](#model-evaluation-and-optimization)*
4. **[Results and Discussion](#results-and-discussion)**
    - 4.1 *[Performance Evaluation](#performance-evaluation)*
    - 4.2 *[Implications and Insights](#implications-and-insights)*
5. **[Conclusion and Recommendations](#conclusion-and-recommendations)**
    - 5.1 *[Summary of Findings](#summary-of-findings)*
    - 5.2 *[Future Work](#future-work)*
6. **[References](#references)**
7. **[Appendices](#appendices)**
    - 7.1 [Code Implementation](#code-implementation)
    - 7.2 [Additional Visualizations and Tables](#additional-visualizations-and-tables)


## <a name="comprehensive-understanding-of-dataset-features"></a>3.1.1 Comprehensive Understanding of Dataset Features

### <a name="data-structure-tables"></a>3.1.1.1 Data Structure Tables

#### <a name="basic-information"></a>3.1.1.1.1 Basic Information

| Prefix            | Data Type   | Null | Description                           | Time Periods List | Data Problems | Preferred Scaling | Preferred Grouping     |
|-------------------|-------------|------|---------------------------------------|-------------------|---------------|-------------------|------------------------|
| N/A               | Numeric     | No   | Identifier                            | N/A               | None          | None              | None                   |
| gil_shihrur       | Numeric     | No   | Age of the individual                 | N/A               | None          | None              | Age groups             |
| kod_min           | Categorical | Yes  | Medical code                          | N/A               | Missing data  | One-hot encoding  | By medical category    |
| kvutza_demografit | Categorical | Yes  | Demographic group                     | N/A               | Missing data  | One-hot encoding  | By demographic         |
| godel_mishpacha   | Numeric     | Yes  | Family size                           | N/A               | Missing data  | Standardization   | By family size         |
| mispar_yeladim    | Numeric     | Yes  | Number of children                    | N/A               | Missing data  | Standardization   | By number of children  |
| zion_sotzio       | Categorical | Yes  | Socioeconomic status                  | N/A               | Missing data  | One-hot encoding  | By status              |
| mekadem           | Categorical | Yes  | Previous treatment status             | N/A               | Missing data  | One-hot encoding  | By treatment status    |
| kod_zakaut_beishpuz| Numeric   | Yes  | Eligibility code for family care      | N/A               | Missing data  | Standardization   | By eligibility         |
| mesheh_ishpuz     | Numeric     | Yes  | Family care budget                    | N/A               | Missing data  | Standardization   | By budget              |
| ishpuzim_num      | Numeric     | Yes  | Number of family care instances       | N/A               | Missing data  | Standardization   | By number of instances |
| zman_bein_ishpuz  | Numeric     | Yes  | Time between family care instances    | N/A               | Missing data  | Standardization   | By time                |


#### <a name="time-based-features-multiple-prefixes"></a>3.1.1.1.2 Time-Based Features (Multiple Prefixes)

| Prefix      | Data Type    | Null | Description                                               | Time Periods List         | Data Problems | Preferred Scaling | Preferred Grouping     |
|-------------|--------------|------|-----------------------------------------------------------|---------------------------|---------------|-------------------|------------------------|
| bkrcnt      | Zero-Inflated| Yes  | Count of bkrcnt type appointments in specific periods     | -12, -9, -6, -3, -1, +1, +3, +6, +9, +12  | Zero-inflated  | Log transformation | By type of appointment |
| bkrcnt_mish | Zero-Inflated| Yes  | Count of bkrcnt_mish type appointments in specific periods | -12, -9, -6, -3, -1, +1, +3, +6, +9, +12  | Zero-inflated  | Log transformation | By type of appointment |
| bkrcnt_mik  | Zero-Inflated| Yes  | Count of bkrcnt_mik type appointments in specific periods  | -12, -9, -6, -3, -1, +1, +3, +6, +9, +12  | Zero-inflated  | Log transformation | By type of appointment |
| bkrcnt_emg  | Zero-Inflated| Yes  | Count of bkrcnt_emg type appointments in specific periods  | -12, -9, -6, -3, -1, +1, +3, +6, +9, +12  | Zero-inflated  | Log transformation | By type of appointment |
| missbkr_cnt | Zero-Inflated| Yes  | Count of missing type appointments in specific periods     | -12, -9, -6, -3, -1, +1, +3, +6, +9, +12  | Zero-inflated  | Log transformation | By type of appointment |

#### <a name="medical-diagnoses-avh-prefix"></a>3.1.1.1.3 Medical Diagnoses (avh prefix)

| Prefix | Data Type | Null | Description                                     | Time Periods List         | Data Problems | Preferred Scaling | Preferred Grouping |
|--------|-----------|------|-------------------------------------------------|---------------------------|---------------|-------------------|--------------------|
| avh    | String    | Yes  | Medical diagnosis in specific periods           | -12, -9, -6, -3, -1, +1, +3, +6, +9, +12 | Missing data    | None              | By diagnosis type  |

#### <a name="types-of-prescriptions-mrs-prefix"></a>3.1.1.1.4 Types of Prescriptions (mrs prefix)


| Prefix | Data Type    | Null | Description                                    | Time Periods List         | Data Problems | Preferred Scaling | Preferred Grouping       |
|--------|--------------|------|------------------------------------------------|---------------------------|---------------|-------------------|--------------------------|
| mrs    | String       | Yes  | Types of prescriptions in specific periods     | -12, -9, -6, -3, -1, +1, +3, +6, +9, +12 | Missing data    | None              | By prescription type     |


## <a name="handling-true-zeros"></a>3.2.1 Handling True Zeros

### Introduction

In our dataset, zeros are not placeholders or missing values; they are "true zeros," indicating the absence of a feature or condition. These zeros are particularly prevalent in time-based features like `bkrcnt`, `bkrcnt_mish`, and `bkrcnt_mik`, where a zero denotes no appointments of a certain type during the specific period. Handling these zeros correctly is crucial for the integrity and interpretability of our predictive models.

### Methods for Handling True Zeros

#### Zero-Inflation Techniques

Given that our dataset is zero-inflated, one approach is to use zero-inflated models. These models have two parts: one part models the count variable using Poisson or Negative Binomial distribution, and the other part models the excess zeros. This approach is implemented in the `data_preeprocess_functions.py` file.

#### Log Transformation

Another approach to manage the skewness introduced by true zeros is log transformation. However, since the log of zero is undefined, we would first replace zeros with a very small positive number before applying the transformation. This approach is also implemented in the `data_preeprocess_functions.py` file.

### Code Snippets

#### Zero-Inflation Technique

```python
# Code from data_preeprocess_functions.py
def log_transform(df, column):
    df[column] = df[column].apply(lambda x: np.log(x if x > 0 else 1e-9))
```

#### Conclusion

Handling true zeros appropriately ensures that our predictive models will be both accurate and interpretable. By leveraging specialized techniques for zero-inflation and transformations, we maintain the integrity of our dataset.

### <a name="addressing-non-normal-distribution"></a>3.2.2 Addressing Non-Normal Distribution

#### Introduction

In machine learning, the assumption of normality for numerical variables can often simplify modeling techniques and increase the predictive power of a model. However, in real-world scenarios, especially in healthcare data like ours, variables often do not follow this distribution. Our dataset has features that are zero-inflated and skewed, challenging the assumptions of many statistical techniques and machine learning models.

#### Importance of Normal Distribution

Non-normal distribution in the predictors can introduce bias and reduce the reliability and interpretability of machine learning models. For instance, many parametric models like linear regression assume that the errors are normally distributed, an assumption that can be violated if the predictors themselves are not normally distributed.

#### Identification Methods

Before applying any transformations, it's crucial to identify which variables do not follow a normal distribution. Techniques such as histogram plotting, Shapiro-Wilk test, or D’Agostino and Pearson’s Test can be employed to statistically measure the normality of the features.

#### Techniques Used for Transformation

To address non-normality, several transformation techniques can be used, depending on the type and extent of skewness:

1. **Log Transformation**: Useful for right-skewed data.
2. **Square/Cube Root Transformation**: Helpful for less skewed data.
3. **Box-Cox Transformation**: A more general form that chooses the best power transformation of the data that reduces skewness.
4. **Min-Max Scaling**: Particularly useful for features that have different ranges.

#### Implementation in Code

In our project, we employed log-transformation and Min-Max scaling after identifying the skewness in the variables. These transformations were incorporated into the pre-processing pipeline to ensure that every feature gets appropriately transformed before model training.

#### Conclusion

Addressing the non-normal distribution is critical for the predictive power and interpretability of our models. By identifying and transforming skewed features, we make our models more robust and reliable, fulfilling the underlying assumptions that many algorithms hold.

## <a name="model-development"></a>3.3 Model Development

### Introduction

The process of model development involves the design and testing of various machine learning algorithms tailored to our dataset. Given the complex nature of our data, including zero-inflated features and non-normal distribution, we employed a two-pronged approach: one pipeline for pre-hospitalization data and another for post-hospitalization data.

### <a name="pipeline-for-pre-hospitalization-period"></a>3.3.1 Pipeline for Pre-Hospitalization Period

#### Objective

The primary aim is to predict the likelihood of suicide attempts before any hospitalization occurs. This is crucial for early intervention and potentially life-saving actions.

#### Model Selection

We chose a zero-inflated classifier tailored for our data's unique characteristics. This model effectively handles the zero-inflation in our features.

#### Feature Engineering

Variables like age, medical codes, and demographic groups were transformed and normalized. Time-based features were log-transformed to address skewness.

#### Implementation in Code

The pipeline integrates data preprocessing, feature selection, and model training into a streamlined workflow. Hyperparameters were optimized using grid search.

### <a name="pipeline-for-post-hospitalization-period"></a>3.3.2 Pipeline for Post-Hospitalization Period

#### Objective

The focus here is to predict the risk of suicide attempts after a hospital stay, facilitating long-term care strategies.

#### Model Selection

Given the changing nature of data post-hospitalization, a different variant of zero-inflated models was used to capture these dynamics.

#### Feature Engineering

Post-hospitalization features were also transformed and normalized, with particular attention to newly emerging variables like types of prescriptions.

#### Implementation in Code

This pipeline similarly encapsulates all steps from data preprocessing to model training, with the ability to update the model as new data comes in.

### <a name="incorporation-of-p-value-wald-test-and-variance-logic"></a>3.3.3 Incorporation of P-value, Wald Test, and Variance Logic

#### Objective

To ensure the statistical significance of our models, we incorporated tests like the P-value and Wald Test.

#### Methods Employed

1. **P-value**: Used to test the null hypothesis and to check the significance of the features.
2. **Wald Test**: Used for hypothesis testing in complex models.
3. **Variance Logic**: To ensure our model's confidence, we examined the variance in the predictions.

#### Implementation in Code

These statistical tests were automated in our pipeline, providing real-time validation of the model's effectiveness.

### Conclusion

The model development phase was meticulously planned and executed to cater to the data's complexities. The pipelines were built to be robust and flexible, capable of adapting to new data, thus making our model both reliable and scalable.







