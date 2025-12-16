#This file works to import and run the evaluation function from evaluate.py
#Importing the necessary libraries
import sys
import os
import importlib

#Add 'src' folder to the path
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.append(os.path.join(project_root, 'src'))


#Importing the evaluation script
import evaluate

#Force reloading the module to get the latest changes
importlib.reload(evaluate)


print(" STARTING EVALUATION SUMMARY ")
print("   - Loading models from 'models/'...")
print("   - Testing on the LAST 15% of data (Strictly Unseen).")
print("   - Comparing Accuracy and picking a winner.\n")

#Run the function
if __name__ == "__main__":
    winner_name, y_true, y_pred = evaluate.evaluate_models()

if winner_name:
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(y_true, y_pred)

        print("\n" + "="*60)
        print("*** WINNER ***")
        print("="*60)
        print(f"\n  Model: {winner_name}")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\n  Saved as: models/buy_sell_classifier.pkl")
        print("\n" + "="*60)
        print("READY FOR DEPLOYMENT!")
        print("="*60)

else:
        print("\nERROR: Evaluation failed")
        print("Run: python src/train.py")

# Step 8 --- Serialize the Model
import joblib
joblib.dump(models, "models/buy_sell_classifier.pkl")
