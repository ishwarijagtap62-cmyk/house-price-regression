# House Price Regression

## Practical Assignment-03: Performance Evaluation of Regression Model

This project implements a **House Price Prediction** system using **Linear Regression** and manually implemented **Gradient Descent**.

The project evaluates the performance of the regression models using **MAE, MSE, RMSE, and R² Score** and analyzes the convergence of Gradient Descent.

---

## Problem Statement

The objective of this practical is to develop a regression model for a real-world application of **house price prediction**.

The model predicts house prices using features such as:

* Area
* Number of bedrooms
* Number of bathrooms
* Age of the property

The practical includes two approaches:

1. Linear Regression using Scikit-learn.
2. Manual Gradient Descent optimization for Linear Regression.

The performance of both approaches is evaluated and compared using regression performance metrics.

---

## Objectives

1. To develop a Linear Regression model for house price prediction.
2. To preprocess and prepare the dataset for regression analysis.
3. To evaluate the regression model using MAE, MSE, RMSE and R² Score.
4. To implement Gradient Descent optimization for Linear Regression.
5. To analyze the convergence and performance of Gradient Descent.

---

## Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**
* **VS Code / Jupyter Notebook / Google Colab**

---

## Dataset

A synthetic but realistic house-price dataset is generated using a fixed random seed for reproducibility.

### Features

| Feature   | Description                   |
| --------- | ----------------------------- |
| Area      | House area in square feet     |
| Bedrooms  | Number of bedrooms            |
| Bathrooms | Number of bathrooms           |
| Age       | Age of the property in years  |
| Price     | House price / target variable |

The dataset contains **300 samples**.

---

## Machine Learning Models

### 1. Linear Regression

Linear Regression is used to predict house prices based on the selected features.

The multiple Linear Regression equation is:

```text
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

Scikit-learn's `LinearRegression` is used for the standard model.

### 2. Gradient Descent

Gradient Descent is manually implemented to optimize the parameters of the Linear Regression model.

The parameter update rule is:

```text
θ = θ - α(∂J/∂θ)
```

where:

* `θ` = model parameter
* `α` = learning rate
* `J` = cost function

Feature scaling is used to achieve smooth convergence.

---

## Performance Metrics

The following metrics are used:

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

**Lower MAE is better.**

### MSE — Mean Squared Error

Measures the average squared prediction error.

**Lower MSE is better.**

### RMSE — Root Mean Squared Error

RMSE is the square root of MSE and is expressed in the same unit as the target variable.

**Lower RMSE is better.**

### R² Score

R² measures the proportion of variation in the target variable explained by the model.

**Higher R² is better.**

---

## Project Workflow

```text
Dataset Generation
        ↓
Data Preprocessing
        ↓
Feature Selection
        ↓
Train-Test Split
        ↓
Linear Regression
        ↓
Manual Gradient Descent
        ↓
Prediction
        ↓
Performance Evaluation
        ↓
Model Comparison
        ↓
Visualization
```

---

## Results

The models are evaluated on the same **80% training and 20% testing split**.

| Model                              |      MAE |          MSE |     RMSE | R² Score |
| ---------------------------------- | -------: | -----------: | -------: | -------: |
| Linear Regression                  | 19367.65 | 587823142.24 | 24245.06 |   0.9590 |
| Gradient Descent Linear Regression | 19367.65 | 587823142.24 | 24245.06 |   0.9590 |

The two approaches produce nearly identical results, indicating that the manually implemented Gradient Descent successfully optimizes the Linear Regression model.

---

## Visualizations

The program generates the following graphs:

### 1. Actual vs Predicted House Price

Shows the relationship between actual house prices and prices predicted by the Linear Regression model.

### 2. Regression Line

Shows the relationship between house area and house price.

### 3. Gradient Descent Cost vs Iterations

Shows how the cost decreases as the number of Gradient Descent iterations increases.

---

## Project Structure

```text
house-price-regression/
│
├── regression_model.py
├── README.md
├── requirements.txt
│
└── screenshots/
    ├── output.png
    ├── metrics.png
    ├── regression_graph.png
    └── gradient_descent.png
```

The Python program also generates:

```text
fig1_actual_vs_predicted.png
fig2_regression_line.png
fig3_cost_vs_iterations.png
```

---

## Installation

Install the required libraries using:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Or use:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/house-price-regression.git
```

### Step 2: Open the project

```bash
cd house-price-regression
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the program

```bash
python regression_model.py
```

---

## Gradient Descent Configuration

The manual Gradient Descent implementation uses:

```text
Learning Rate = 0.05
Iterations = 1000
```

The features are standardized before applying Gradient Descent.

The cost decreases rapidly during the initial iterations and then stabilizes, indicating convergence.

---

## Conclusion

This project demonstrates the application of **Linear Regression** to a real-world house price prediction problem.

The model is evaluated using **MAE, MSE, RMSE and R² Score**. Manual Gradient Descent is also implemented to understand iterative optimization and convergence.

Both Linear Regression and Gradient Descent produce nearly identical performance on the test dataset, demonstrating the correctness of the Gradient Descent implementation.

---


