from sklearn.tree import DecisionTreeClassifier
import joblib
import os
from data import load_data, split_data

def fit_model(X_train, y_train, feature_names):
    dt = DecisionTreeClassifier(max_depth=None, random_state=12)
    dt.fit(X_train, y_train)

    os.makedirs("../model", exist_ok=True)
    joblib.dump(dt, "../model/wine_model.pkl")

    print("Feature importances:")
    for name, importance in zip(feature_names, dt.feature_importances_):
        print(f"{name}: {importance:.4f}")


if __name__ == "__main__":
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    fit_model(X_train, y_train, feature_names)
