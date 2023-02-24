import pandas as pd
import numpy as np
import os
import sys
import glob

from sklearn.linear_model import BayesianRidge

# Import GridSearch_module
module_folder = 'Tune_Parameters'
sys.path.insert(0, module_folder)
import GridSearch
from GridSearch import run_grid_search

# Import Train_test_split
sys.path.append(module_folder)
from SplitData import data_split

# import Fillna combine
sys.path.append('Data_fillna')
from fillna import FillnaCombine


# Flag for Find_Economic
find_Economic = False
# Save all Economic_program_name in one list
All_file_name = []
# Create a DataFrame for ours fillna data
# time_index = pd.date_range("2015-01-01", "2021-12-31", freq='D')
# df = pd.DataFrame(index = time_index)
# Coice Economic_prgram

files = glob.glob('Data\\fillna\\*.csv')
for file in files:
    file_name = os.path.basename(file)
    file_name = file_name.split("_")[0]
    All_file_name.append(file_name)
    
# Number each option(Economic_program)
NumberList = [(i, Economic) for i, Economic in enumerate(All_file_name, 1)]
print(NumberList)
Economic_program = input("Please Enter your choice(Enter a number)")


for economic_number in NumberList:
    # If find the Economial_program
    if economic_number[0] == int(Economic_program):
        # Give it the name of the economic variable
        Economic_program = economic_number[1]
        find_Economic = True
        break
if find_Economic == True:
    pass
# Exit the program
else:
    print("The number you entered is not in the list")
    sys.exit()

# Segment the populated data before standardizing

# Fillna Combine
# df are fillna combine data
df =  FillnaCombine()
# Let's predict the target in the last position(convenient value selection)
cols = list(df.columns)
cols.append(cols.pop(cols.index(f'{Economic_program}_Price')))
df = df.reindex(columns = cols)

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
# Set the y_pred index from y_test.index

# Create a DataFrame for y_pred
ComparitionTable = pd.DataFrame(data = {'y_test':y_test, f"Predict_the_price_of_{economi_project_name}":y_pred}, index=y_test.index, )
# inverse Normalization

print(ComparitionTable)