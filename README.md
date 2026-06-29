# Task-number-1

# Task 1: Data Cleaning & Preprocessing
# Titanic Dataset

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1. Load Dataset
# ------------------------------

# Load Titanic dataset
df = pd.read_csv("titanic.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# ------------------------------
# 2. Handle Missing Values
# ------------------------------

# Fill missing Age values with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked values with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column because too many missing values
df.drop(columns=['Cabin'], inplace=True)

# Verify missing values again
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ------------------------------
# 3. Convert Categorical Data
# ------------------------------

# Convert Sex column to numerical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encoding for Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\nDataset After Encoding:")
print(df.head())

# ------------------------------
# 4. Normalize / Standardize Data
# ------------------------------

from sklearn.preprocessing import StandardScaler

# Select numerical columns
num_cols = ['Age', 'Fare']

# Standardize features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nStandardized Data:")
print(df[num_cols].head())

# ------------------------------
# 5. Visualize Outliers
# ------------------------------

plt.figure(figsize=(10, 5))

# Boxplot for Age
plt.subplot(1, 2, 1)
sns.boxplot(y=df['Age'])
plt.title("Boxplot of Age")

# Boxplot for Fare
plt.subplot(1, 2, 2)
sns.boxplot(y=df['Fare'])
plt.title("Boxplot of Fare")

plt.tight_layout()
plt.show()

# ------------------------------
# 6. Remove Outliers (IQR Method)
# ------------------------------

# Function to remove outliers
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return data[(data[column] >= lower) & (data[column] <= upper)]

# Remove outliers from Fare
df = remove_outliers(df, 'Fare')

print("\nShape After Removing Outliers:")
print(df.shape)

# ------------------------------
# Final Cleaned Dataset
# ------------------------------

print("\nFinal Cleaned Dataset:")
print(df.head())

# Save cleaned dataset
df.to_csv("cleaned_titanic.csv", index=False)

print("\nCleaned dataset saved as 'cleaned_titanic.csv'")
