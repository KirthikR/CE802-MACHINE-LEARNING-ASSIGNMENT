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
# Project Title

Short description of the project.

## Table of Contents

- [Data Loading and Exploration](#data-loading-and-exploration)
- [Data Preprocessing](#data-preprocessing)
- [Data Splitting](#data-splitting)
- [Model Training - Linear Regression](#model-training---linear-regression)
- [Model Testing and Prediction](#model-testing-and-prediction)
- [Explanation](#explanation)
- [Next Steps](#next-steps)

## Data Loading and Exploration

- The dataset "CE802_P3_Data.csv" was loaded into a Pandas DataFrame named `data_p3`.
- Initial exploration involved examining the first six rows of the dataset using `data_p3.head()` to understand its structure.
- Data types were checked using `data_p3.dtypes`, general information about the dataset was obtained using `data_p3.info()`, and missing values were identified using `data_p3.isnull().sum()`.

## Data Preprocessing

- Categorical variables F6 and F10 were encoded using one-hot encoding with `pd.get_dummies`.
- Encoded variables were concatenated with the original dataset to create `data_p3_edited`.
- Original categorical columns (F6 and F10) were dropped from the dataset to ensure compatibility with machine learning algorithms.

## Data Splitting

- Features and the target variable were separated from the edited dataset to create `data_p3_edited_features` and `data_p3_edited_outcome`, respectively.
- The data was split into training and testing sets using `train_test_split` function from scikit-learn.

## Model Training - Linear Regression

- A linear regression model (`lm_reg_model`) was created using scikit-learn's `LinearRegression`.
- The model was trained on the training data using `lm_reg_model.fit`.

## Model Testing and Prediction

- The test dataset "CE802_P3_Test.csv" was loaded into `data_p3_test`.
- Similar preprocessing steps as the training set were applied to prepare the test data.
- The trained linear regression model was used to predict the target variable on the test set, resulting in `lm_reg_pred`.

## Explanation

- **Data Exploration:** This step ensures a thorough understanding of the dataset's structure, data types, and presence of missing values, providing a foundation for subsequent analysis.
- **Data Preprocessing:** Categorical variables are encoded to enable machine learning models to process them effectively. Dropping original categorical columns ensures streamlined data for model compatibility.
- **Model Training:** Linear regression, chosen for its simplicity and interpretability, is trained on the prepared data to establish patterns between features and target variable.
- **Model Testing:** The trained model is evaluated on unseen data to assess its predictive performance and generalization capabilities.
- **Next Steps:** Evaluation metrics such as Mean Squared Error (MSE) or R-squared could be employed to gauge model performance. Further optimization or exploration of alternative algorithms may be considered based on project requirements and model performance.

