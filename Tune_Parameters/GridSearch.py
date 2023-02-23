from sklearn.model_selection import GridSearchCV

def run_grid_search(model, param_grid, X, y):
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
    grid_search.fit(X,y)
    return grid_search.best_estimator_