import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# STEP 1: LOAD DATASET
# =========================================================
df = pd.read_csv("student_performance_updated_1000.csv")

# =========================================================
# STEP 2: BASIC EDA (Before Cleaning)
# =========================================================

print("\n===== BASIC EDA BEFORE CLEANING =====\n")

print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nDatatypes:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary Statistics:\n", df.describe())
print("\nFirst 5 Rows:\n", df.head())

print("\n-------------------------------------------\n")

# =========================================================
# STEP 3: DATA CLEANING
# =========================================================

# Remove rows where StudentID is null
df = df[df['StudentID'].notnull()]

# Fill categorical columns with mode
df['Name'] = df['Name'].fillna(df['Name'].mode()[0])
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['ExtracurricularActivities'] = df['ExtracurricularActivities'].fillna(df['ExtracurricularActivities'].mode()[0])
df['ParentalSupport'] = df['ParentalSupport'].fillna(df['ParentalSupport'].mode()[0])
df['Online Classes Taken'] = df['Online Classes Taken'].fillna(df['Online Classes Taken'].mode()[0])

# Fill numeric columns with mean
df['AttendanceRate'].fillna(df['AttendanceRate'].mean(), inplace=True)
df['StudyHoursPerWeek'].fillna(df['StudyHoursPerWeek'].mean(), inplace=True)
df['PreviousGrade'].fillna(df['PreviousGrade'].mean(), inplace=True)
df['FinalGrade'].fillna(df['FinalGrade'].mean(), inplace=True)
df['Study Hours'].fillna(df['Study Hours'].mean(), inplace=True)
df['Attendance (%)'].fillna(df['Attendance (%)'].mean(), inplace=True)

# Convert StudentID to integer
df['StudentID'] = df['StudentID'].astype(int)

print("\n===== AFTER CLEANING =====\n")
print(df.isnull().sum())

print("\n-------------------------------------------\n")

# =========================================================
# STEP 4: BASIC EDA AFTER CLEANING
# =========================================================

print("\n===== BASIC EDA AFTER CLEANING =====\n")
print("Shape:", df.shape)
print("\nDatatypes:\n", df.dtypes)
print("\nSummary Statistics:\n", df.describe())

# =========================================================
# STEP 5: VISUAL EDA
# =========================================================

# 1. Gender Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', data=df)
plt.title("Gender Count")
plt.show()

# 2. Study Hours Histogram
plt.figure(figsize=(6,4))
sns.histplot(df['Study Hours'], kde=True)
plt.title("Study Hours Distribution")
plt.show()

# 3. Attendance vs Final Grade
plt.figure(figsize=(6,4))
sns.scatterplot(x='AttendanceRate', y='FinalGrade', data=df)
plt.title("Attendance vs Final Grade")
plt.show()

# 4. Correlation Heatmap (Fix)
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# =========================================================
# STEP 6: SAVE CLEANED DATASET
# =========================================================

df.to_csv("cleaned_student_performance.csv", index=False)

