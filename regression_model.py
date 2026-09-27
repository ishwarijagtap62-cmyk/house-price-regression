# =========================================================
# Application: House Price Prediction using Regression
# =========================================================
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
# ---------------------------------------------------------
# Step 1: Load / Create the dataset
# ---------------------------------------------------------
# NOTE: A synthetic but realistic house-price dataset is generated here
# (fixed random seed for reproducibility) since it keeps the practical
# fully self-contained and runnable without an external file.
# The same code works unchanged on any real CSV dataset with the same
# column names (area, bedrooms, bathrooms, age, price).
 
np.random.seed(42)
n_samples = 300
 
area = np.random.randint(500, 4000, n_samples)          # sq. ft.
bedrooms = np.random.randint(1, 6, n_samples)            # count
bathrooms = np.random.randint(1, 4, n_samples)           # count
age = np.random.randint(0, 30, n_samples)                # years
 
noise = np.random.normal(0, 25000, n_samples)
 
price = (50000
         + 120 * area
         + 15000 * bedrooms
         + 10000 * bathrooms
         - 800 * age
         + noise)
 
data = pd.DataFrame({
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age": age,
    "price": price
})
 
print("First 5 rows of the dataset:")
print(data.head())
 
# ---------------------------------------------------------
# Step 2: Basic preprocessing
# ---------------------------------------------------------
print("\nMissing values in each column:")
print(data.isnull().sum())
 
X = data[["area", "bedrooms", "bathrooms", "age"]].values
y = data["price"].values
 
# ---------------------------------------------------------
# Step 3: Train-test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
# ---------------------------------------------------------
# Step 4: Train standard Linear Regression (sklearn)
# ---------------------------------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
 
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)
 
print("\n--- Linear Regression (sklearn) Results ---")
print(f"MAE  : {mae_lr:.2f}")
print(f"MSE  : {mse_lr:.2f}")
print(f"RMSE : {rmse_lr:.2f}")
print(f"R2   : {r2_lr:.4f}")
 
# ---------------------------------------------------------
# Step 5: Manual Gradient Descent implementation
# ---------------------------------------------------------
# Feature scaling is required for Gradient Descent to converge smoothly
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
def gradient_descent(X, y, learning_rate=0.05, iterations=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0.0
    cost_history = []
 
    for i in range(iterations):
        y_pred = np.dot(X, weights) + bias
        error = y_pred - y
 
        # Cost function (Mean Squared Error / 2)
        cost = np.mean(error ** 2) / 2
        cost_history.append(cost)
 
        # Gradients
        dw = (1 / m) * np.dot(X.T, error)
        db = (1 / m) * np.sum(error)
 
        # Parameter update
        weights -= learning_rate * dw
        bias -= learning_rate * db
 
    return weights, bias, cost_history
 
weights, bias, cost_history = gradient_descent(
    X_train_scaled, y_train, learning_rate=0.05, iterations=1000
)
 
y_pred_gd = np.dot(X_test_scaled, weights) + bias
 
mae_gd = mean_absolute_error(y_test, y_pred_gd)
mse_gd = mean_squared_error(y_test, y_pred_gd)
rmse_gd = np.sqrt(mse_gd)
r2_gd = r2_score(y_test, y_pred_gd)
 
print("\n--- Gradient Descent Linear Regression Results ---")
print(f"MAE  : {mae_gd:.2f}")
print(f"MSE  : {mse_gd:.2f}")
print(f"RMSE : {rmse_gd:.2f}")
print(f"R2   : {r2_gd:.4f}")
print(f"Final Cost (last iteration): {cost_history[-1]:.2f}")
 
# ---------------------------------------------------------
# Step 6: Graph 1 - Actual vs Predicted values
# ---------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred_lr, color="blue", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="red", linestyle="--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Price (Linear Regression)")
plt.tight_layout()
plt.savefig("fig1_actual_vs_predicted.png", dpi=150)
plt.close()
 
# ---------------------------------------------------------
# Step 7: Graph 2 - Regression line (Area vs Price)
# ---------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.scatter(data["area"], data["price"], color="green", alpha=0.5, label="Actual data")
sorted_idx = np.argsort(data["area"].values)
area_sorted = data["area"].values[sorted_idx]
# simple 1-feature regression line for visualization only
simple_lr = LinearRegression().fit(data[["area"]], data["price"])
plt.plot(area_sorted, simple_lr.predict(area_sorted.reshape(-1, 1)),
          color="red", linewidth=2, label="Regression line")
plt.xlabel("Area (sq. ft.)")
plt.ylabel("Price")
plt.title("Regression Line: Area vs Price")
plt.legend()
plt.tight_layout()
plt.savefig("fig2_regression_line.png", dpi=150)
plt.close()
 
# ---------------------------------------------------------
# Step 8: Graph 3 - Gradient Descent Cost vs Iterations
# ---------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.plot(range(len(cost_history)), cost_history, color="purple")
plt.xlabel("Iterations")
plt.ylabel("Cost (J)")
plt.title("Gradient Descent: Cost vs Iterations")
plt.tight_layout()
plt.savefig("fig3_cost_vs_iterations.png", dpi=150)
plt.close()
 
# ---------------------------------------------------------
# Step 9: Final comparison
# ---------------------------------------------------------
print("\n--- Comparison Table ---")
print(f"{'Model':35s}{'MAE':>10s}{'MSE':>15s}{'RMSE':>12s}{'R2':>10s}")
print(f"{'Linear Regression':35s}{mae_lr:10.2f}{mse_lr:15.2f}{rmse_lr:12.2f}{r2_lr:10.4f}")
print(f"{'Gradient Descent Regression':35s}{mae_gd:10.2f}{mse_gd:15.2f}{rmse_gd:12.2f}{r2_gd:10.4f}")
