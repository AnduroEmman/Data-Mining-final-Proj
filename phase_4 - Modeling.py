import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load preprocessed datasets (no index column)
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').iloc[:, 0].values
y_test = pd.read_csv('y_test.csv').iloc[:, 0].values

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

trained_models = {}

def evaluate_model(name, model):
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_r2 = r2_score(y_train, train_preds)
    mae = mean_absolute_error(y_test, test_preds)
    mse = mean_squared_error(y_test, test_preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, test_preds)

    print(f"\nModel: {name}")
    print(f"Train R² Score: {train_r2:.3f}")
    print(f"Test MAE: {mae:.3f}")
    print(f"Test MSE: {mse:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test R² Score: {r2:.3f}")

    joblib.dump(model, f"{name.replace(' ', '_').lower()}_model.pkl")
    trained_models[name] = model

for name, model in models.items():
    evaluate_model(name, model)

# Select best model based on test R² score
best_name = max(trained_models, key=lambda name: r2_score(y_test, trained_models[name].predict(X_test)))
best_model = trained_models[best_name]
best_score = r2_score(y_test, best_model.predict(X_test))

joblib.dump(best_model, "best_model.pkl")
print(f"\nBest Model: {best_name} with R² Score = {best_score:.3f}")
