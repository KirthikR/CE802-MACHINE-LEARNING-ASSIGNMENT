# CE802-MACHINE-LEARNING-ASSIGNEMNT

# CODE 1
# Machine Learning Project: Binary Classification with Python

This Python script is part of a machine learning project aimed at solving a binary classification problem. The objective is to predict the target variable labeled "Class" based on a set of input features labeled F1 to F19.

## Overview

The project involves the following key steps:

1. **Data Loading and Preprocessing:**
   - Import necessary libraries and load the training and test datasets (`CE802_P2_Data.csv` and `CE802_P2_Test.csv`) using pandas.
   - Check for missing values in the training dataset and drop the column "F20" which contains missing values.
   - Split the datasets into features (X) and target variable (y), and handle missing values by replacing them with the mean of the respective feature using an imputer.

2. **Exploratory Data Analysis (EDA):**
   - Compute a correlation matrix to examine the relationships between features. This helps in understanding feature importance and potential multicollinearity issues.

3. **Model Building:**
   - Employ three different classifiers: Decision Tree, Random Forest, and Gradient Boosting Machine (GBM).
   - Train each classifier on the training data after handling missing values and evaluate their performance using accuracy score on the test data.

4. **Model Evaluation:**
   - Print out the accuracy scores of each model and visualize them using a heatmap.
   - Utilize GridSearchCV to tune hyperparameters for the Gradient Boosting Machine classifier to improve its performance.

5. **Test Data Prediction:**
   - Select the best model (GBM) and train it on the entire training dataset.
   - Use the trained model to predict the target variable for the test dataset.
   - Add the predicted labels to the test dataset, and save the modified dataset as a CSV file (`CE802_P2_Test.csv`).

## Conclusion

This script demonstrates a typical workflow for a supervised machine learning classification problem. It covers data loading, preprocessing, model training, evaluation, and prediction. Additionally, it incorporates techniques for handling missing values and hyperparameter tuning to improve model performance.



# CODE 2
# Project Overview:
# 1. Data Loading and Exploration:

You started by loading the dataset (CE802_P3_Data.csv) into a Pandas DataFrame named data_p3.
Checked the first six rows of the dataset using data_p3.head() to understand its structure.
Explored data types, information, and checked for missing values using data_p3.dtypes, data_p3.info(), and data_p3.isnull().sum().
# 2. Data Preprocessing:

Encoded categorical variables F6 and F10 using one-hot encoding with pd.get_dummies.
Concatenated the encoded variables with the original dataset (data_p3_edited).
Dropped the original categorical columns (F6 and F10) from the dataset.
# 3. Data Splitting:

Separated the features (data_p3_edited_features) and the target variable (data_p3_edited_outcome) from the edited dataset.
Split the data into training and testing sets using train_test_split.
# 4. Model Training - Linear Regression:

Created a linear regression model (lm_reg_model) using scikit-learn's LinearRegression.
Trained the model on the training data with lm_reg_model.fit.
# 5. Model Testing and Prediction:

Loaded the test dataset (CE802_P3_Test.csv) into data_p3_test.
Applied the same preprocessing steps as in the training set to prepare the test data.
Used the trained linear regression model to predict the target variable on the test set (lm_reg_pred).

# Explanation:
# Data Exploration:

You examined the dataset to understand its structure, checked data types, and ensured there were no missing values.
# Data Preprocessing:

Categorical variables were one-hot encoded to make them compatible with machine learning models.
The dataset was modified to include these encoded variables and exclude the original categorical columns.
# Model Training:

You chose a linear regression model for the task, initialized it, and trained it using the training dataset.
# Model Testing:

The trained model was then used to make predictions on a separate test dataset.
# Next Steps:

Depending on the project requirements, you might want to evaluate the model's performance using metrics like Mean Squared Error (MSE) or R-squared.
Further optimization of the model or exploration of other algorithms could be considered.
