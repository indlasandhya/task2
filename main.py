from sklearn.model_selection import train_test_split

from dataset_load import load_dataset
from feature_scaling import scale_features
from train_models import train_linear, train_ridge, train_tree
from evaluation import evaluate_model
from visualization import plot_results

import pandas as pd

print("Loading Dataset...")
df = load_dataset()

X = df.drop("TargetPrice", axis=1)
y = df["TargetPrice"]

print("Splitting Dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Scaling Features...")
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

print("\nTraining Models...")

linear_model = train_linear(X_train_scaled, y_train)
ridge_model = train_ridge(X_train_scaled, y_train)
tree_model = train_tree(X_train, y_train)

print("\nEvaluating Models...")

rmse_lr, r2_lr = evaluate_model(linear_model, X_test_scaled, y_test)
rmse_ridge, r2_ridge = evaluate_model(ridge_model, X_test_scaled, y_test)
rmse_tree, r2_tree = evaluate_model(tree_model, X_test, y_test)

results = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression", "Decision Tree"],
    "RMSE": [rmse_lr, rmse_ridge, rmse_tree],
    "R2 Score": [r2_lr, r2_ridge, r2_tree]
})

print("\nModel Performance Comparison:\n")
print(results)

best = results.sort_values(by="RMSE").iloc[0]
print("\nBest Model:", best["Model"])

print("\nPlotting Best Model Predictions...")
predictions = linear_model.predict(X_test_scaled)
plot_results(y_test, predictions)
