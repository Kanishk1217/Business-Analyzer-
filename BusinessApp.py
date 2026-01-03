import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
# Ensure 'ai_job_market.csv' is in the same directory as this script
df = pd.read_csv('ai_job_market.csv')

# Display initial data
print("First 5 rows of the dataset:")
print(df.head())

# Information and Calculation of our dataset
print("\nInformation of our dataset:")
df.info()

print("\nCalculation of our data set:")
print(df.describe())

print("\nNumbers of columns, Number of rows:")
print(df.shape)

# Detailed data types and Null values
print("\nData type of column:")
print(df.dtypes)

print("\nNull value in each feature:")
print(df.isnull().sum())

# Column Categorization
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("\nNumeric Columns:", numeric_cols)
print("Categorical Columns:", categorical_cols)

# Summaries
if numeric_cols:
    print("\nNumeric Summary")
    print(df[numeric_cols].describe())

if categorical_cols:
    print("\nCategorical Summary")
    for col in categorical_cols:
        print(f"\nColumn: {col}")
        print(df[col].value_counts().head(5))

# Visualization: Distributions and Outliers
for col in numeric_cols:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(df[col], kde=True, ax=ax[0])
    ax[0].set_title(f'Distribution of {col}')
    
    sns.boxplot(x=df[col], ax=ax[1])
    ax[1].set_title(f'Outliers in {col}')
    
    plt.show()

# Prediction Evaluation Logic 
# (Note: These variables y_test, y_pred, target_col, and problem_type 
# need to be defined in your training script for this part to execute)
"""
if 'problem_type' in locals():
    if problem_type == "regression":
        plt.figure(figsize=(6,6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{target_col}: Actual vs Predicted")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.show()
        
    elif problem_type == "classification":
        sns.countplot(x=y_pred)
        plt.title(f"{target_col} Predicted Class Distribution")
        plt.show()
"""
