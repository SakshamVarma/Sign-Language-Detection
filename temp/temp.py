import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the pre-trained model
model = load_model('sign_language_model_data_aug_new.h5')  # Replace with your model's path

# Define the labels for the ASL signs
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
          'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the BGR frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        # Get hand landmarks for the first detected hand
        landmarks = results.multi_hand_landmarks[0].landmark
        
        # Extract x, y coordinates of the wrist
        wrist_x = int(landmarks[0].x * frame.shape[1])
        wrist_y = int(landmarks[0].y * frame.shape[0])
        
        # Calculate bounding box coordinates for cropping
        box_size = 140
        top_left_x = max(0, wrist_x - box_size // 2)
        top_left_y = max(0, wrist_y - box_size // 2)
        bottom_right_x = min(frame.shape[1], wrist_x + box_size // 2)
        bottom_right_y = min(frame.shape[0], wrist_y + box_size // 2)
        
        # Crop and resize hand region
        hand_region = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        resized_hand = cv2.resize(hand_region, (28, 28))
        gray_hand = cv2.cvtColor(resized_hand, cv2.COLOR_BGR2GRAY)
        
        # Preprocess the hand image
        normalized_hand = gray_hand / 255.0
        reshaped_hand = normalized_hand.reshape(1, 28, 28, 1)
        
        # Make prediction
        prediction = model.predict(reshaped_hand)
        predicted_label = labels[np.argmax(prediction)]
        
        # Display the results on the frame
        cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame with hand tracking
    cv2.imshow('ASL Sign Prediction', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
