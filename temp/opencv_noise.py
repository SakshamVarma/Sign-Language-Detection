import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
from keras.models import load_model

# Load the trained model
model = load_model('sign_language_model_data_aug_new.h5')

# Create a list to map predicted labels to sign language symbols
alphabet_symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open a connection to the webcam (0 for default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Convert normalized landmarks to pixel coordinates
            height, width, _ = frame.shape
            landmarks = [(int(l.x * width), int(l.y * height)) for l in hand_landmarks.landmark]

            # Calculate bounding box around landmarks with extra padding
            padding = 30  # Adjust the padding as needed
            min_x = max(0, min(landmarks, key=lambda p: p[0])[0] - padding)
            max_x = min(width, max(landmarks, key=lambda p: p[0])[0] + padding)
            min_y = max(0, min(landmarks, key=lambda p: p[1])[1] - padding)
            max_y = min(height, max(landmarks, key=lambda p: p[1])[1] + padding)

            # Calculate new bounding box coordinates to maintain a square aspect ratio
            max_dim = max(max_x - min_x, max_y - min_y)
            new_min_x = max(0, min_x - (max_dim - (max_x - min_x)) // 2)
            new_max_x = min(width, max_x + (max_dim - (max_x - min_x)) // 2)
            new_min_y = max(0, min_y - (max_dim - (max_y - min_y)) // 2)
            new_max_y = min(height, max_y + (max_dim - (max_y - min_y)) // 2)

            if new_max_x > new_min_x and new_max_y > new_min_y:
                # Crop the hand area
                cropped_hand = frame[new_min_y:new_max_y, new_min_x:new_max_x]

                # Convert cropped image to grayscale
                gray_hand = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2GRAY)

                # Resize the grayscale hand for model input
                resized_hand = cv2.resize(gray_hand, (28, 28))

                # Normalize the image
                normalized_hand = resized_hand / 255.0
                input_data = normalized_hand.reshape(1, 28, 28, 1)

                # Predict the label
                prediction = model.predict(input_data)
                predicted_label = np.argmax(prediction)
                predicted_symbol = alphabet_symbols[predicted_label]

                # Display the predicted symbol on the image
                cv2.putText(frame, f'Predicted: {predicted_symbol}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the grayscale hand image
                cv2.imshow('Grayscale Hand', resized_hand)

    # Display the camera image
    cv2.imshow('Camera and Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
