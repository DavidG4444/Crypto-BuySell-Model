## Step 7 --- Evaluation
#Importing all the necessary libraries
import pandas as pd
import numpy as np
import os
import joblib
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Converting the processed training data into a dataframe
training_data = pd.read_csv(r"data/processed/training_data.csv")
training_data.head()
training_data.info()

def evaluate_models():
    current_script_dir = os.path.dirname(os.path.abspath(file))
    project_root = os.path.dirname(current_script_dir)

    data_path = os.path.join(project_root, 'data', 'labeled', 'labeled_data.csv')
    model_dir = os.path.join(project_root, 'saved_models')

#Checking to see that the folder exists
    if not os.path.exists(model_dir):
        print(f"⚠️  Models folder not found at {model_dir}. Creating it now...")
        os.makedirs(model_dir)
        print(f"✅ Created folder: {model_dir}")
        print("ℹ️  Please add your trained .pkl models here and rerun the evaluation.")
        return None, None, None

    print(" Starting Model Evaluation Arena...")

    # --- 2. PREPARE TEST DATA ---
    if not os.path.exists(data_path):
        print(f" Error: Data not found at {data_path}")
        return None, None, None

    df = pd.read_csv(data_path)
    df.dropna(inplace=True)

    drop_cols = ['open_time', 'close_time', 'ignore', 'future_return', 'label', 'threshold_buy', 'threshold_sell']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df['label']

    test_start = int(len(df) * 0.85)
    X_test = X.iloc[test_start:]
    y_test = y.iloc[test_start:]

    print(f"   Testing on {len(X_test)} rows (Last 15%)\n")

    # --- 3. EVALUATION LOOP ---
    best_acc = -1
    best_model_name = None
    best_preds = None

    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and f != 'best_crypto_model.pkl']

    if not model_files:
        print("⚠️  No model files found in saved_models/. Please add your trained .pkl models and rerun.")
        return None, None, None

    print(f"{'MODEL':<20} | {'ACCURACY':<10}")
    print("-" * 35)

    for filename in model_files:
        model_name = filename.replace('.pkl', '')
        model_path = os.path.join(model_dir, filename)

        try:
            model = joblib.load(model_path)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            print(f"{model_name:<20} | {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_model_name = model_name
                best_preds = preds

        except Exception as e:
            print(f" Error loading {model_name}: {e}")

    # --- 4. SHOW WINNER ---
    if best_model_name is None:
        print(" No valid models could be evaluated.")
        return None, None, None

    print("-" * 35)
    print(f" WINNER: {best_model_name} (Accuracy: {best_acc:.4f})")

    # --- 5. SAVE BEST MODEL ---
    src_file = os.path.join(model_dir, f"{best_model_name}.pkl")
    dst_file = os.path.join(model_dir, "best_crypto_model.pkl")
    try:
        shutil.copyfile(src_file, dst_file)
        print(f" Copied winner to: {dst_file}")
    except Exception as e:
        print(f" Could not copy best model: {e}")

    # --- 6. TEXT REPORT ---
    print("\n Classification Report (Winner):")
    print(classification_report(y_test, best_preds, target_names=['SELL', 'HOLD', 'BUY']))

    return best_model_name, y_test, best_preds

if name == "main":
    winner_name, y_true, y_pred = evaluate_models()

    if winner_name is not None:
        print("\n Generating Confusion Matrix Plot...")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Pred SELL', 'Pred HOLD', 'Pred BUY'],
                    yticklabels=['Act SELL', 'Act HOLD', 'Act BUY'])
        plt.title(f'Confusion Matrix: {winner_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

#Saving the models into one folder.
import os
import joblib

Absolute path to project root
project_root = os.path.dirname(os.path.abspath("."))  # adjust if needed
models_folder = os.path.join(project_root, "saved_models")
os.makedirs(models_folder, exist_ok=True)

Save models
joblib.dump(rf_model,  os.path.join(models_folder, "Random_Forest_Model.pkl"))
joblib.dump(lgbm_model, os.path.join(models_folder, "LightGBM_Model.pkl"))
joblib.dump(xgb_model,  os.path.join(models_folder, "XGBoost_Model.pkl"))
joblib.dump(cat_model,  os.path.join(models_folder, "CatBoost_Model.pkl"))

Confirm saved files
print("✅ Models saved in:", models_folder)
print(os.listdir(models_folder))


#Evaluation code.
1. SETUP & IMPORTS
import sys
import os
import importlib

Add 'src' to the path
sys.path.append(os.path.abspath('../src'))

Import the evaluation script
import evaluate

Force reload
importlib.reload(evaluate)

2. EXECUTE EVALUATION ARENA
print(" STARTING EVALUATION ARENA...")
print("   - Loading models from 'models/'...")
print("   - Testing on the LAST 15% of data (Strictly Unseen).")
print("   - Comparing Accuracy and picking a winner.\n")

Run the function
evaluate.evaluate_models()

# Step 8 --- Serialize the Model
import joblib
joblib.dump(model, "models/buy_sell_classifier.pkl")
