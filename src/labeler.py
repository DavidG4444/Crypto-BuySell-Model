#Step 4 --- Label Generation (Target Variable)
#Importing libraries
import os
import pandas as pd

#Importing data into df form from csv form
df = pd.read_csv(r"C:\Users\USER\OneDrive\Desktop\LuxDev DSA\Capstone-Project\Crypto-BuySell-Model\data\processed\BTCUSDT_1dmodified.csv")
df.info()
# Create reference to DataFrame

def save_processed_csv(df, symbol, interval):

#Save a DataFrame into data/raw/(symbol_interval).csv
    os.makedirs("data/processed", exist_ok=True)
    file_path = f"data/processed/{symbol}_{interval}.csv"
    df.to_csv(file_path, index=False)
    print(f"Saved: {file_path}")

save_processed_csv(df, "BTCUSDT", "1dmodified2")

df = pd.read_csv("data/processed/BTCUSDT_1dmod.csv")
df.info()

#Creating target labels based on future returns
df["future_return"] = df["close"].pct_change().shift(-1)

def label(row):
    if row["future_return"] > 0.02:
        return 2
    elif row["future_return"] < -0.02:
        return 0
    else:
        return 1

df["label"] = df.apply(label, axis=1)


#Function to save and store csv file of data obtained
def save_processed_csv(df, symbol, interval):

#Save a DataFrame into data/raw/(symbol_interval).csv
    os.makedirs("data/processed", exist_ok=True)
    file_path = f"data/processed/{symbol}_{interval}.csv"
    df.to_csv(file_path, index=False)
    print(f"Saved: {file_path}")

save_processed_csv(df, "BTCUSDT", "1dmodified")


## Step 5 --- Train/Test Split

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

#Check for NaN values
print(f"NaN values before cleaning:\n{df.isna().sum()}")

#Drop NaN rows
df_clean = df.dropna()

print(f"\nOriginal data: {len(df)} rows")
print(f"After dropping NaN: {len(df_clean)} rows")
df_clean.info()


#Plotting a Feature Correlation Matrix to determine what columns to use for the model
numeric_cols = [
    'future_return', 'label', 'rsi', 'macd', 'macd_hist',
    'sma_20', 'sma_50', 'sma_200', 'volatility', 'volume'
]

#Filtering the columns in the dataframe
existing_cols = [col for col in numeric_cols if col in df_clean.columns]

plt.figure(figsize=(12, 10))
sns.heatmap(df_clean[existing_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix')
plt.show()

#Define features and target

target_col = "label"  #column with your Buy/Sell/Hold labels
feature_cols = df_clean.columns.difference([target_col, "future_return", "open_time", "close_time"])

X = df_clean[feature_cols]
y = df_clean[target_col]


#Train/Validation/Test split

train_size = int(len(df_clean) * 0.70)
val_size = int(len(df_clean) * 0.15)
test_start = train_size + val_size

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

X_val = X.iloc[train_size:test_start]
y_val = y.iloc[train_size:test_start]

X_test = X.iloc[test_start:]
y_test = y.iloc[test_start:]


#Print summary

print(f"Train: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"Validation: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
print(f"Test: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")


## Step 6 --- Model Training
#Importing all the necessary libraries
import xgboost
import tensorflow
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

#A function to calculate and print all evaluation metrics
def calculate_metrics(y_true, y_pred, model_name, dataset_name):

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n{model_name} - {dataset_name} Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

#Peforming Model Training with various models

#Logistic Regression Model
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight='balanced')
model_lr.fit(X_train, y_train)


#Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

#Train
rf_model.fit(X_train, y_train)

#Predict
rf_train_pred = rf_model.predict(X_train)
rf_val_pred = rf_model.predict(X_val)
rf_test_pred = rf_model.predict(X_test)

#Calculating the metrics
rf_train_metrics = calculate_metrics(y_train, rf_train_pred, "Random Forest", "Training")
rf_val_metrics = calculate_metrics(y_val, rf_val_pred, "Random Forest", "Validation")
rf_test_metrics = calculate_metrics(y_test, rf_test_pred, "Random Forest", "Test")

#Printing the Classification Report for the Test Set
print("\n" + "="*60)
print("RANDOM FOREST - TEST SET CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, rf_test_pred))

#LightGBM Model
lgbm_model = LGBMClassifier(
    n_estimators=200,
    max_depth=15,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

#Train
lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='multi_logloss'
)

#Predictions
lgbm_train_pred = lgbm_model.predict(X_train)
lgbm_val_pred = lgbm_model.predict(X_val)
lgbm_test_pred = lgbm_model.predict(X_test)

#LightGBM Metrics
lgbm_train_metrics = calculate_metrics(y_train, lgbm_train_pred, "LightGBM", "Training")
lgbm_val_metrics = calculate_metrics(y_val, lgbm_val_pred, "LightGBM", "Validation")
lgbm_test_metrics = calculate_metrics(y_test, lgbm_test_pred, "LightGBM", "Test")

#XGBoost Model
model = XGBClassifier()
model.fit(X_train, y_train)


#CatBoost Model
model_cat = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    random_seed=42,
    verbose=0
)

#Train
model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

#Predict
y_val_pred = model_cat.predict(X_val)

#Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, digits=4))

df_clean.info()
