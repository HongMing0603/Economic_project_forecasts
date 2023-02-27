import pandas as pd
import numpy as np
import os
import sys
import glob

from sklearn.svm import SVR

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

# import Normalization module
sys.path.append('Data_Normalization_split')
from normalization import Normalization_afterSplit, Denormalize

# import Validation_index
sys.path.append('Validation_index')
from vd_index import rmse, mape, smape, r2, MAE, Each_error_value

# import model
from sklearn.ensemble import RandomForestRegressor

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
Economic_program = input("Please Enter your choice(Enter a number):")

for economic_number in NumberList:
    # (1,Bitcoin)
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
y = df.iloc[:,-1:]

# Split Train and Test
X_train, X_test, y_train, y_test = data_split(X,y)

# Normalization Dataset
X_train_scalered = Normalization_afterSplit("X_scaler", X_train, "train")
X_test_scalered = Normalization_afterSplit("X_scaler", X_test, "test")
y_train_scalered = Normalization_afterSplit("y_scaler", y_train, "train")

# Create a model
model = RandomForestRegressor()

param_grid = {"n_estimators": [50, 100, 150],
            "max_depth": [5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
            }


# Shows how much time the model tooks
# Get the model name using the py file
model_path = os.path.abspath(__file__)
model_name = os.path.basename(model_path)
# Get the model name without .py
model_name = model_name.split(".")[0]

# Parameters adjusted
# The bestmodel here is your model instance that you can use directly to predict
# Pass Economic_name for GridSearch (Economic_program)
best_model = run_grid_search(model, param_grid, X_train_scalered, y_train_scalered, Economic_program, model_name)

# Forecasting Values
y_pred = best_model.predict(X_test_scalered) 

# Get ours predict economi project name
economi_project_name = df.columns.values[-1]
economi_project_name = economi_project_name.split('_')[0]

# Create a dataFrame for y_pred 
y_pred = pd.DataFrame(data=y_pred, index=y_test.index, columns=["y_Pred"])

# Denormalize it (y_pred)
y_pred = Denormalize(y_pred)
y_pred = pd.DataFrame(data=y_pred, columns=["y_Pred"])
y_pred.index = y_test.index

print(y_pred.columns)
combine_df = pd.concat([y_test, y_pred], axis=1)
combine_df.columns.values[0] = "y_Test"
combine_df.columns.values[1] = "y_Pred"


# Calculate the error values
Each_error_value(model_name, combine_df["y_Test"], combine_df["y_Pred"])
