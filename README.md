# EHR-PREDICTIVE-MODELING
Tools for doing predictive modeling on electronic medical records data

# Predictive Modeling with LASSO Logistic Regression
This repository hosts the code used for performing predicitve modeling on electronic medical records data in Python. We use logistic regression with L1 (LASSO) penalty to select important features from lab expression levels data, e.g. proteomics and metabolomics data, to explore how well these analytes predict a medical outcome. Meanwhile, we fit a logistic regression with only the baseline demographic and clinical data to understand how well the baseline data predicts a medical outcome. Then, we combine the baseline data with expression data and fit another logistic regression to see how much predictive power it gained by adding the analytes into the equation.

Optionally, we can choose the number of analytes that survived LASSO according to specific needs. Here, we give the example of choosing to have five analytes to be included in the final model.
( Optionally, a “top-5 variables” plus “aggregate variables”)

## Table of Contents
- [Introduction](#introduction)
- [Requirements and Setup](#requirements-and-setup)
- [Data Preparation](#data-preparation)
- [Predictive Modeling Workflow](#predictive-modeling-workflow)
- [Running the Code](#running-the-code)
- [Interpreting Results](#interpreting-results)
- [Support](#support)


## Introduction
Our goal is to predict **adverse liver events** (or **Hepatocellular Carcinoma Flag**) using clinical and lab data. We have:
- **Proteomics data**: A pandas DataFrame called `xpr`.
- **Metabolomics data**: Another DataFrame called `lab_xpr` (with or without “aggregate” variables).

We perform:
1. **Cross-validation** over a grid of L1 penalty strengths to pick the best penalty for logistic regression.
2. **Top-5 variable selection** for the non-aggregate variables subset.
3. **Optional aggregator step**: The top 5 variables combined with aggregate variable columns.
(add tips for reducing run-time, also trade-offs)

## Requirements and Setup
- **Python** 3.8+  ????
- Common libraries:
  - `pandas`  
  - `numpy`  
  - `scipy`  
  - `scikit-learn`  
  - `statsmodels`
  - etc.........




## Data Preparation
(data source:   link here)


(data types requirements)


(data transformation -------- with regards to model assumptions)
We transformed baseline clinical variables to approximate a normal or less skewed distribution in order to align with analyses assumptions. Skewness was computed before and after each transformation to assess improvement in distribution symmetry.  Histograms were generated to visually confirm these distributional changes. The final transformed variables were used in downstream analyses, ensuring more robust parameter estimation and reducing potential biases from heavily skewed data. And the transformation deployed were: 

AGE: no transformation

SEX: 'Female' and 'Male', use 'Female' as reference 

RACE: "WHITE","BLACK OR AFRICAN AMERICAN","ASIAN","AMERICAN INDIAN OR ALASKA NATIVE", use 'WHITE' as reference group,

BMI: base 2 logarithm 

Albumin (g/dL): natural log of [6 – raw value] 

Bilirubin (g/dL): base 2 logarithm of [raw value +1] 

Creatinine (g/dL): base 2 logarithm of [raw value +1] 

FIB-4 score: base 2 logarithm of [raw value +1] 

Hemoglobin A1C (%): Winsorization (capping extreme values) of the highest 3% of values 


## Predictive Modeling Workflow
We will have some helper functions, and a risk model function. After we've had our output, we can choose to plot ROC curves.

We examined three primary modeling scenarios: 
- Proteomics-only 
- Metabolomics-only 
- Proteomics + Metabolomics 

For each scenario, we compared three models: 
a) A baseline clinical model (encompassing AGE, SEX, RACE, BMI, and hemoglobin A1C), 
b）An analytes-only model, 
c) A combined model (incorporating the baseline clinical variables plus cross-validated analytes-based predictions). 

#### Analytes-Only Model 

We trained the analytes-only model with LASSO-penalized logistic regression, varying the regularization strength (C) from 0.025 to 1.975 in increments of 0.025. Model performance was assessed using leave-one-out cross-validation, during which the area under the ROC curve (AUC) was calculated on held-out subsets to identify the optimal penalty parameter. The model was then retrained on the entire dataset with the best penalty value, and variable importance was determined by standardized coefficients (raw coefficients divided by the standard deviation of the corresponding variable). The top 10 predictors, ranked by absolute value of standardized coefficient, were reported in bar charts and spreadsheets. We utilized a bootstrap approach (100 iterations) to estimate 95% confidence intervals for the AUC, thereby providing an indication of the model’s performance stability. 

#### Baseline-Only and Combined Models 

To facilitate comparisons, the baseline-only model was fitted using logistic regression on the six baseline clinical predictors, and the combined model integrated these clinical variables with cross-validated predictions from the analytes-only model. We again computed AUC values by comparing predicted probabilities with observed outcomes, employing bootstrapping to obtain 95% confidence intervals. Likelihood ratio tests were performed to evaluate the incremental contribution of the combined model over the baseline model, thus determining whether the inclusion of proteomic/metabolomic data statistically significantly improved predictive accuracy. 

#### (optional)ROC Curve Generation               **need consistency**

We plotted ROC curves for all three modeling approaches to illustrate the trade-off between the true positive rate (sensitivity) and false positive rate (1 − specificity). There are four curves: one analytes-only curve, and three cruves for the three combined model. Curves were annotated with corresponding AUC metrics, enabling a visual assessment of each model’s discriminatory power. 

### helper Functions
### Risk Model
(**repos to where they are stored**) + explanation


## Running the Code
(details on how to use the functions to produce results: models, expression data matrices)


## Interpreting Results




## Support
source of data
liabilities??
contact info



##
##
