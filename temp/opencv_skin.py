import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained ASL model
model = load_model('sign_language_model_data_aug_new.h5')  # Load your adapted model here

# Create a list to map predicted labels to alphabet symbols
alphabet_symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z']

# Open a connection to the webcam (0 for default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame for model input
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (28, 28))  # Resize to model's input size
    normalized_frame = resized_frame / 255.0  # Normalize pixel values

    input_data = normalized_frame.reshape(1, 28, 28, 1)

    # Predict the label
    prediction = model.predict(input_data)
    predicted_label = np.argmax(prediction)
    predicted_symbol = alphabet_symbols[predicted_label]

    # Display the predicted symbol on the image
    cv2.putText(frame, f'Predicted: {predicted_symbol}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the camera image with prediction
    cv2.imshow('ASL Sign Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
