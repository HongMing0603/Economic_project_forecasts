from sklearn.model_selection import GridSearchCV
import time

def run_grid_search(model, param_grid, X_train, y_train):
    """
    Arg:
        model:Your model
        param_grid:You select the range to adjust for the model's parameter
        X_train
        y_train
    return:
        grid_search.best_estimator:Instantiation of optimal model parameters
        time_cost:How much time it took to find the parameters
    """
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
    start = time.time()
    grid_search.fit(X_train,y_train)
    end = time.time()
    # Calculate the time it takes
    time_cost = end-start
    return grid_search.best_estimator_, time_cost