# CE802-MACHINE-LEARNING-ASSIGNEMNT

# CODE 1
# Data Setup and Exploration:

Loaded the training data and checked for any missing values.

# Data Cleaning:

Removed columns with missing data to ensure a clean dataset for analysis.

# Correlation Analysis:

Explored relationships between features by visualizing a correlation matrix.

# Handling Missing Values:

Imputed missing values using the average, ensuring completeness for further analysis.

# Model Training and Evaluation:

Trained Decision Tree and Random Forest models, assessing their predictive accuracy.

# Optimizing Model Performance:

Used GridSearchCV to find the best settings for a Gradient Boosting Machine model.

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
