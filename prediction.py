import pandas as pd
import pickle
from evaluation import ml_scores

X_valid = pd.read_csv('X_valid.csv')
y_valid = pd.read_csv('y_valid.csv')

# Load the trained model from the pickle file
with open('final_model.pkl', 'rb') as f:
    final_model = pickle.load(f)

# Make predictions using the trained model
y_pred = final_model.predict(X_valid)

xgb_results = ml_scores('XGBoost', y_valid, y_pred)
print(xgb_results)

predictions_df = pd.DataFrame({'Actual': y_valid['isFraud'], 'Predicted': y_pred})
predictions_df.to_csv('predictions.csv', index=False)

# Print evaluation metrics
print("Evaluation Metrics:")
print(f"Balanced Accuracy: {xgb_results.loc['XGBoost', 'Balanced Accuracy']}")
print(f"Precision: {xgb_results.loc['XGBoost', 'Precision']}")
print(f"Recall: {xgb_results.loc['XGBoost', 'Recall']}")
print(f"F1 Score: {xgb_results.loc['XGBoost', 'F1']}")
print(f"Cohen's Kappa: {xgb_results.loc['XGBoost', 'Kappa']}")