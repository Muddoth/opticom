import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor

model = None

def calibrate(iris_data, screen_data):
    global model
    X = np.array(iris_data)
    y = np.array(screen_data)
    model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=500)
    model.fit(X, y)
    with open("eye_tracking_model.pkl", "wb") as f:
        pickle.dump(model, f)
    return "Model trained successfully."

def predict(iris_position):
    global model
    if not model:
        with open("eye_tracking_model.pkl", "rb") as f:
            model = pickle.load(f)
    prediction = model.predict([iris_position])
    return prediction.tolist()
