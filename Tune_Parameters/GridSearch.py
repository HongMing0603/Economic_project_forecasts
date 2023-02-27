from sklearn.model_selection import GridSearchCV
import time
import joblib
import sys
import os

def run_grid_search(model, param_grid, X_train, y_train, economic_program, model_name):
    """
    Arg:
        model:Your model
        param_grid:You select the range to adjust for the model's parameter
        X_train
        y_train
        economic_program: Help us determine which economic variable to use when exporting the model
        model_name: if you use BayesianRidgeRegression you just type his name
    return:
        grid_search.best_estimator:Instantiation of optimal model parameters
        time_cost:How much time it took to find the parameters
    """

    model_folder = "Tune_Parameters/GridSearchModel/"
    grid_search_model_name = f"{model_name}_GridSearch_{economic_program}"
    # Determine whether the model exists
    best_model = Determine_model(model_folder, grid_search_model_name)
    if best_model != "":
        # Already GridSearch
        # Display the search time
        
        # Shows how much time the parameter search
        # took and show best parameters
        elapsed_parameters(model_name, best_model)

        return best_model
    else:
        grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
        start = time.time()
        grid_search.fit(X_train,y_train)
        end = time.time()
        # Calculate the time it takes
        elapsed_time = end-start
        # best model 
        best_model = grid_search
        # Write the time record into it as well
        best_model.elapsed_time = elapsed_time
        # Shows how much time the parameter search
        # took and show best parameters
        elapsed_parameters(model_name, best_model)

        # Save Best Parameters model
        
        # Judgement the folder exist or not
        # Assume the folder does not exist
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        else:
            print("The folder already exists, come on!!")


        joblib.dump(best_model, model_folder + grid_search_model_name)
        return best_model

def Determine_model(model_folder, grid_search_model_name):
    """
    Determine whether model exists 
    """
    # best_model_file
    best_model_file = model_folder + grid_search_model_name
    # Determine whether best_model_exists
    if os.path.exists(best_model_file):
        # exists
        # Directory return file without grid_search(Because GridSearch compleate)
        best_model = joblib.load(best_model_file)
        return best_model
    else:
        return ""

def elapsed_parameters(model_name, best_model):
    """
    Displays the parameter search time and the best parameters
    """
    print(f"{model_name}_Model parameter tunning took these times:{best_model.elapsed_time} seconds")
    # Display the best model and parameters
    best_parameters = best_model.best_params_
    print(f"{model_name}'s best parameters are: {best_parameters}")