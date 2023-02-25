from sklearn.model_selection import GridSearchCV

def run_grid_search(model, param_grid, X_train, y_train):
    """
    Arg:
        model:Your model
        param_grid:You select the range to adjust for the model's parameter
        X_train
        y_train
    """
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train,y_train)
    return grid_search.best_estimator_