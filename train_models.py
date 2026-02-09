from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

def train_linear(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_ridge(X_train, y_train):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model

def train_tree(X_train, y_train):
    model = DecisionTreeRegressor(max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model
