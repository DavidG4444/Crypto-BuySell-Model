#Step 5 --- Train/Test Split

#Importing all the necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df= pd.read_csv(r"C:\Users\USER\OneDrive\Desktop\LuxDev DSA\Capstone-Project\Crypto-BuySell-Model\data\processed\BTCUSDT_1dmodified.csv")
df.info()


#Checking for and dropping all rows with NaN rows
# Check for NaN values
print(f"NaN values before cleaning:\n{df.isna().sum()}")

# Drop NaN rows
df_clean = df.dropna()

print(f"\nOriginal data: {len(df)} rows")
print(f"After dropping NaN: {len(df_clean)} rows")

df_clean.info()



#Plotting a Feature Correlation Matrix to determine what columns to use for the model
numeric_cols = [
    'future_return', 'label', 'rsi', 'macd', 'macd_hist',
    'sma_20', 'sma_50', 'sma_200', 'volatility', 'volume'
]
# Filtering columns
existing_cols = [col for col in numeric_cols if col in df_clean.columns]

plt.figure(figsize=(12, 10))
sns.heatmap(df_clean[existing_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix')
plt.show()



# Performing a Train/Validation/Test Split
# Define features and target
target_col = "label"  #column with your Buy/Sell/Hold labels
feature_cols = df_clean.columns.difference([target_col, "future_return", "open_time", "close_time"])

X = df_clean[feature_cols]
y = df_clean[target_col]

# Train/Validation/Test split
train_size = int(len(df_clean) * 0.70)
val_size = int(len(df_clean) * 0.15)
test_start = train_size + val_size

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

X_val = X.iloc[train_size:test_start]
y_val = y.iloc[train_size:test_start]

X_test = X.iloc[test_start:]
y_test = y.iloc[test_start:]

# Printing a summary of the data after the split
print(f"Train: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"Validation: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
print(f"Test: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")




#Step 6 --- Model Training

#Importing all the necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
