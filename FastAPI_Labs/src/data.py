import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load the Wine dataset and return as a pandas DataFrame.
    """
    wine = load_wine()
    indexes = [6, 9, 12]  # flavanoids, color_intensity, proline
    X = wine.data[:, indexes]
    y = wine.target

# Create DataFrame with matching column names
    selected_columns = [wine.feature_names[i] for i in indexes]
    df = pd.DataFrame(X, columns=selected_columns)
    df['target'] = y

# Print first 5 rows
    print("First 5 rows of the dataset:")
    print(df.tail())
    

    return X, y, selected_columns


def split_data(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=12
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X, y, selected_columns = load_data()
