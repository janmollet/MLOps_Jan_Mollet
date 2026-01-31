import joblib

model = joblib.load("C:/Users/jan10/OneDrive/Escritorio/Escriptori/NORTHEASTERN/MLOps/MLOps/Labs/API_Labs/FastAPI_Labs/model/wine_model.pkl")


# Try distinct examples
X_test = [
    [3.0, 4.0, 520.0],   # Barolo
    [1.0, 3.0, 750.0],   # Grignolino
    [4.5, 10.0, 1000.0]  # Barbera
]

for x in X_test:
    print(model.predict([x]))
