from sklearn.model_selection import train_test_split

def data_split(X, y):
    """
    X: array-like
    y: array-like
    """
    test_size = float(input("Please enter your test size(Like 0.2):"))
    X_train, X_test, y_train, y_test = trainTest_split(X, y, testSize=test_size)
    return X_train, X_test, y_train, y_test

def trainTest_split(X, y, testSize=0.2, shuffle=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, shuffle=shuffle)
    return X_train, X_test, y_train, y_test