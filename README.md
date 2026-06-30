# AI & ML Internship - Task 1: Data Cleaning & Preprocessing

## Objective
Perform data cleaning and preprocessing on the Titanic dataset.

## Tools Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Steps Performed
- Loaded the Titanic dataset
- Explored the dataset using head(), info(), describe()
- Checked missing values
- Filled missing values:
  - Age → Median
  - Embarked → Mode
- Removed the Cabin column due to excessive missing values
- Encoded categorical columns using Label Encoding
- Standardized numerical features (Age and Fare)
- Visualized outliers using boxplots
- Removed outliers using the IQR method
- Saved the cleaned dataset

## Files
- Task1_Data_Cleaning_Preprocessing.ipynb
- cleaned_titanic.csv
- README.md

## Outcome
Successfully cleaned and preprocessed the Titanic dataset for machine learning applications.
