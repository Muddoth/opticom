import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from sklearn.neural_network import MLPRegressor
import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")


# Initialize components
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False

# Variables for calibration
calibration_data = {"relative_iris": [], "screen": []}  # Store input-output pairs
model = None  # Placeholder for ANN model


# Load model if already trained
try:
    with open("eye_tracking_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded.")
except FileNotFoundError:
    print("No trained model found. Please calibrate and train.")

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        #right iris drawing
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)    

            cv2.circle(frame, (x, y), 2, (255, 0, 154), -1)
        
        key = cv2.waitKey(1) & 0xFF  # Capture the key press

        if key == ord('c'):  # Calibration mode

            # iris coordinates
            iris_x = landmark.x * frame_w
            iris_y = landmark.y * frame_h

            # Record the current mouse position on the screen
            mouse_x, mouse_y = pyautogui.position()                

            # Append relative iris data
            calibration_data["relative_iris"].append([iris_x, iris_y])

            # Append corresponding screen coordinates
            calibration_data["screen"].append({"mouse_x": mouse_x, "mouse_y": mouse_y})

            print(f"Iris ({iris_x}, {iris_y}), Mouse ({mouse_x}, {mouse_y})")

            
        elif key == ord('t'):  # Train the model
            if len(calibration_data["relative_iris"]) > 5:  # Ensure sufficient data
                # Prepare input data (iris positions)
                X = np.array(calibration_data["relative_iris"])  # Inputs: Relative Iris positions

                # Prepare output data (screen coordinates)
                y = np.array([[point["mouse_x"], point["mouse_y"]] for point in calibration_data["screen"]])

                # Train the model
                model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=500)
                model.fit(X, y)

                # Save the trained model
                with open("eye_tracking_model.pkl", "wb") as f:
                    pickle.dump(model, f)
                print("Model trained and saved.")
            else:
                print("Not enough data to train. Please calibrate more points.")


        elif key == ord('q'): # quit
            break
            
        # Predict cursor position using the model
        if model:
            # Calculate normalized iris positions relative to the frame
            input_x = landmark.x * frame_w
            input_y = landmark.y * frame_h
            screen_coords = model.predict([[input_x, input_y]])[0]
            mouse_x = screen_coords[0] 
            mouse_y = screen_coords[1] 
            pyautogui.moveTo(mouse_x, mouse_y)  
            print(f"Iris ({input_x}, {input_y}), Mouse ({mouse_x}, {mouse_y})")    



        # Detect a blink by tracking eyelid landmarks
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        if (left[0].y - left[1].y) < 0.004:
            #pyautogui.click()
            pyautogui.sleep(1)

    cv2.imshow('Opticom', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
