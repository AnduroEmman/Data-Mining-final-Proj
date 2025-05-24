import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load test data without index
X_test = pd.read_csv('X_test.csv')
y_test_df = pd.read_csv('y_test.csv')
if y_test_df.shape[1] == 1:
    y_test = y_test_df.iloc[:, 0].values
else:
    raise ValueError("y_test.csv should have exactly one column.")

# Load the best saved model
best_model = joblib.load('best_model.pkl')

# Predict on test set
predictions = best_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("Phase 5: Evaluation Results for Best Model")
print(f"Test MAE: {mae:.2f}")
print(f"Test MSE: {mse:.2f}")
print(f"Test RMSE: {rmse:.2f}")
print(f"Test R² Score: {r2:.2f}")

# Actual vs Predicted scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.grid(True)
plt.tight_layout()
plt.show()

# Residuals distribution plot (optional)
plt.figure(figsize=(8, 6))
sns.histplot(y_test - predictions, kde=True)
plt.title("Residuals Distribution")
plt.xlabel("Residuals (Actual - Predicted)")
plt.show()

# Sample predictions table
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions,
    'Difference': y_test - predictions
})

print("\nSample prediction results (first 10 rows):")
print(results_df.head(10))
