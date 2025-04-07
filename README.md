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


## Predictive Modeling Workflow
We will have some helper functions, and a risk model function. After we've had our output, we can choose to plot ROC curves.
### helper Functions
### Risk Model
### ROC curves


## Running the Code
(details on how to use the functions to produce results)


## Interpreting Results




## Support
source of data
liabilities??
contact info



##
##
