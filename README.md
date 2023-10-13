# suecide_attrpt_predictions

# Table of Contents

1. **Introduction**
   1.1 *Background and Motivation*
   1.2 *Objectives*
   1.3 *Significance of the Project*

2. **Literature Review**
   2.1 *Supervised Learning*
   2.2 *Zero-Inflated Data with true zero meaning*
       2.2.1 Zero-Inflated Data
       2.2.2 Data with lare percentily of meaningfull zero
   2.3 *Potability Prediction in Pandemic Scenarios*

4. **Methodology**
   3.1 *Data Acquisition and Description*
       3.1.1 Dataset Features
       3.1.2 Zero-Inflation in Dataset Features
   3.2 *Data Pre-processing*
       3.2.1 Handling True Zeros
       3.2.2 Addressing Non-Normal Distribution
   3.3 *Model Development*
       3.3.1 Pipeline for Pre-Hospitalization Period
       3.3.2 Pipeline for Post-Hospitalization Period
       3.3.3 Incorporation of P-value, Wald Test, and Variance Logic
   3.4 *Model Evaluation and Optimization*

5. **Results and Discussion**
   4.1 *Performance Evaluation*
   4.2 *Implications and Insights*

6. **Conclusion and Recommendations**
   5.1 *Summary of Findings*
   5.2 *Future Work*

7. **References**

8. **Appendices**
   7.1 Code Implementation
   7.2 Additional Visualizations and Tables





## Comprehensive Understanding of Dataset Features

### Data Structure Tables

#### Basic Information

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





#### Time-Based Features (Multiple Prefixes)

| Prefix      | Data Type    | Null | Description                                               | Time Periods List         | Data Problems | Preferred Scaling | Preferred Grouping     |
|-------------|--------------|------|-----------------------------------------------------------|---------------------------|---------------|-------------------|------------------------|
| bkrcnt      | Zero-Inflated| Yes  | Count of bkrcnt type appointments in specific periods     | -12, -9, -6, -3, -1, +1, +3, +6, +9, +12  | Zero-inflated  | Log transformation | By type of appointment |
| bkrcnt_mish | Zero-Inflated| Yes  | Count of bkrcnt_mish type appointments in specific periods | -12, -9, -6, -3, -1, +1, +3, +6, +9, +12  | Zero-inflated  | Log transformation | By type of appointment |
| bkrcnt_mik  | Zero-Inflated| Yes  | Count of bkrcnt_mik type appointments in specific periods  | -12, -9, -6, -3, -1, +1, +3, +6, +9, +12  | Zero-inflated  | Log transformation | By type of appointment |
| bkrcnt_emg  | Zero-Inflated| Yes  | Count of bkrcnt_emg type appointments in specific periods  | -12, -9, -6, -3, -1, +1, +3, +6, +9, +12  | Zero-inflated  | Log transformation | By type of appointment |
| missbkr_cnt | Zero-Inflated| Yes  | Count of missing type appointments in specific periods     | -12, -9, -6, -3, -1, +1, +3, +6, +9, +12  | Zero-inflated  | Log transformation | By type of appointment |



#### Medical Diagnoses (avh prefix)

| Prefix | Data Type | Null | Description                                     | Time Periods List         | Data Problems | Preferred Scaling | Preferred Grouping |
|--------|-----------|------|-------------------------------------------------|---------------------------|---------------|-------------------|--------------------|
| avh    | String    | Yes  | Medical diagnosis in specific periods           | -12, -9, -6, -3, -1, +1, +3, +6, +9, +12 | Missing data    | None              | By diagnosis type  |

#### Types of Prescriptions (mrs prefix)

| Prefix | Data Type    | Null | Description                                    | Time Periods List         | Data Problems | Preferred Scaling | Preferred Grouping       |
|--------|--------------|------|------------------------------------------------|---------------------------|---------------|-------------------|--------------------------|
| mrs    | String       | Yes  | Types of prescriptions in specific periods     | -12, -9, -6, -3, -1, +1, +3, +6, +9, +12 | Missing data    | None              | By prescription type     |

