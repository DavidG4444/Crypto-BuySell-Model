# Step 9 --- Prediction Pipeline
import joblib
import pandas as pd
import numpy as np
import os

    #model = joblib.load("models/buy_sell_classifier.pkl")
    #return model.predict(features)

def predict(features):
#Making predictions using the trained model
#predictions: array of predictions (0=SELL, 1=HOLD, 2=BUY)

    model_path = "models/buy_sell_classifier.pkl"

    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            "Please run: python src/train.py"
        )

    # Load model
    model = joblib.load(model_path)

    # Make predictions
    predictions = model.predict(features)

    return predictions

if __name__ == "__main__":
    print("="*60)
    print("CRYPTOCURRENCY PREDICTION")
    print("="*60)

    # Example usage
    print("Loading data...")
    df = pd.read_csv("data/processed/training_data.csv")

    # Get features
    drop_cols = ['label', 'future_return', 'open_time', 'close_time',
                 'ignore', 'threshold_buy', 'threshold_sell']
    feature_cols = [col for col in df.columns if col not in drop_cols]

    # Get last 5 samples
    X = df[feature_cols].tail(50)

    print(f"\nMaking predictions on {len(X)} samples...")
    predictions = predict(X)

    # Display results
    class_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

    print("="*60)
    print("RESULTS")
    print("="*60)

    for i, pred in enumerate(predictions, 1):
        # ✅ PROPER WAY: Use .item() for numpy scalars
        if isinstance(pred, np.ndarray):
            pred_value = pred.item()  # Extract scalar from 0-d array
        elif hasattr(pred, 'item'):
            pred_value = pred.item()  # For numpy types like np.int64
        else:
            pred_value = int(pred)    # For regular Python int

        action = class_names[pred_value]
        print(f"  Sample {i}: {action} (class {pred_value})")

    print("\n" + "="*60)
    print("PREDICTION COMPLETE")
    print("="*60)
