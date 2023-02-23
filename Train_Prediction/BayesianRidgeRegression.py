import pandas as pd
import numpy as np
import os
import sys

from sklearn.linear_model import BayesianRidge

# Import GridSearch_module
module_folder = 'Tune_Parameters'
sys.path.insert(0, module_folder)
import GridSearch
from GridSearch import run_grid_search

# Import Train_test_split
sys.path.append(module_folder)
from SplitData import data_split

# Import Normalization Data
df = pd.read_csv('Data\\Normalization\\NormalizationData.csv')
df = df.set_index("Date")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split Train and Test
X_train, X_test, y_train, y_test = data_split(X,y)
# Create a model
model = BayesianRidge()
param_grid = {
    "alpha_1":[1e-5, 1e-6, 1e-7],
    "alpha_2":[1e-5, 1e-6, 1e-7],
    "lambda_1":[1e-5, 1e-6, 1e-7],
    "lambda_2":[1e-5, 1e-6, 1e-7],
}
# Parameters adjusted
# The bestmodel here is your model instance that you can use directly to predict
best_model = run_grid_search(model, param_grid, X_train, y_train)

# Forecasting Values
y_pred = best_model.predict(X_test)

# Get ours predict economi project name
economi_project_name = df.columns.values[-1]
economi_project_name = economi_project_name.split('_')[-1]
# Create a DataFrame for y_pred
y_pred = pd.DataFrame(data = y_pred, index=y_test.index, columns=[f"Predict_the_price_of_{economi_project_name}"])
print(y_pred)
