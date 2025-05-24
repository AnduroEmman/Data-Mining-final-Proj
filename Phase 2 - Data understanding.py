# phase2_data_understanding.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "CarPrice_Assignment.csv"
df = pd.read_csv(file_path)

# 1. Preview Data
print("=== Dataset Head ===")
print(df.head())

print("\n=== Dataset Shape ===")
print(df.shape)

print("\n=== Dataset Info ===")
print(df.info())

# 2. Describe Data (Numerical & Categorical)
print("\n=== Numerical Summary ===")
print(df.describe())

print("\n=== Categorical Columns Summary ===")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nUnique values in '{col}': {df[col].nunique()}")
    print(df[col].value_counts())

# 3. Exploratory Data Analysis (EDA)

# Numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# 3.1 Histograms for numeric columns
df[numeric_cols].hist(bins=20, figsize=(15, 10))
plt.suptitle("Histograms of Numeric Features")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 3.2 Boxplots for numeric columns to detect outliers
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(4, 5, i)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.suptitle("Boxplots of Numeric Features", y=1.02)
plt.show()

# 3.3 Correlation matrix heatmap
plt.figure(figsize=(12, 8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# 3.4 Top correlations with price
corr_target = corr['price'].abs().sort_values(ascending=False)
print("\n=== Top Correlations with Price ===")
print(corr_target)

# === Important Plots for Report ===

# 1. Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], bins=30, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# 2. Price vs Engine Size
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='enginesize', y='price')
plt.title('Scatter Plot: Engine Size vs Price')
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.show()

# 3. Price vs Curb Weight
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='curbweight', y='price')
plt.title('Scatter Plot: Curb Weight vs Price')
plt.xlabel("Curb Weight")
plt.ylabel("Price")
plt.show()

# 4. Price vs Horsepower
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='horsepower', y='price')
plt.title('Scatter Plot: Horsepower vs Price')
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.show()

# 5. Price by Fuel Type
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x='fueltype', y='price')
plt.title('Boxplot: Price by Fuel Type')
plt.xlabel("Fuel Type")
plt.ylabel("Price")
plt.show()

# 6. Car Body Type Count
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='carbody')
plt.title("Count of Car Body Types")
plt.xlabel("Car Body")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.show()

# 7. Price by Car Body
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='carbody', y='price')
plt.title("Boxplot: Price by Car Body")
plt.xlabel("Car Body")
plt.ylabel("Price")
plt.xticks(rotation=30)
plt.show()

# 8. Price by All Categorical Features (Looped)
for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x=col, y='price')
    plt.title(f'Price Distribution by {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 4. Check for Missing Values
print("\n=== Missing Values ===")
print(df.isnull().sum())

# 5. Check for zero or suspicious values in numeric columns
print("\n=== Zero or Negative Value Checks in Numeric Columns ===")
for col in numeric_cols:
    zero_count = (df[col] == 0).sum()
    neg_count = (df[col] < 0).sum()
    print(f"{col}: zeros = {zero_count}, negatives = {neg_count}")

# 6. Skewness of price
print("\nSkewness of price:", df['price'].skew())

# 7. Summary of observations for report
summary = """
Phase 2 Summary:
- Dataset has {} rows and {} columns.
- Contains {} numeric features and {} categorical features.
- No missing values detected.
- Some numeric features have zero values which may need investigation.
- Correlation matrix shows strong correlation between price and features like horsepower, curbweight, and engine size.
- Potential outliers detected in features like 'price', 'horsepower', and 'curbweight' via boxplots.
- Categorical features such as fuel type and car body show variance in price distribution across groups.
- Price distribution is right-skewed.
- Scatter plots suggest nonlinear relationships between price and key numeric predictors.
- Further data cleaning and feature engineering needed in Phase 3.
""".format(df.shape[0], df.shape[1], len(numeric_cols), len(categorical_cols))

print(summary)
