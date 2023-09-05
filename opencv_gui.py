import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
from keras.models import load_model
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Load the trained model
model = load_model('sign_language_model_data_aug_new3.h5')

# Create a list to map predicted labels to sign language symbols
alphabet_symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Function to process each frame and update the GUI
def update_frame():
    ret, frame = cap.read()

    if ret:
        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
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

                    # Calculate the width and height of the bounding box
                    box_width = max_x - min_x
                    box_height = max_y - min_y

                    # Determine the maximum dimension for a square aspect ratio
                    max_dim = max(box_width, box_height)

                    # Calculate new bounding box coordinates to maintain a square aspect ratio
                    new_min_x = max(0, min_x - (max_dim - box_width) // 2)
                    new_max_x = min(width, max_x + (max_dim - box_width) // 2)
                    new_min_y = max(0, min_y - (max_dim - box_height) // 2)
                    new_max_y = min(height, max_y + (max_dim - box_height) // 2)

                    # Ensure the new bounding box has non-zero dimensions
                    if new_max_x > new_min_x and new_max_y > new_min_y:
                        # Crop the hand area
                        cropped_hand = frame[new_min_y:new_max_y, new_min_x:new_max_x]

                        # Convert cropped image to grayscale
                        gray_hand = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2GRAY)

                        # Resize the cropped hand for model input (to 28x28)
                        resized_hand = cv2.resize(gray_hand, (28, 28))

                        # Create a white canvas with the same size as the resized hand
                        white_background = np.ones_like(resized_hand) * 255

                        # Paste the resized hand onto the white background
                        white_background[0:28, 0:28] = resized_hand

                        # Normalize the image
                        normalized_hand = white_background / 255.0
                        input_data = normalized_hand.reshape(1, 28, 28, 1)

                        # Predict the label
                        prediction = model.predict(input_data)
                        predicted_label = np.argmax(prediction)
                        predicted_symbol = alphabet_symbols[predicted_label]

                        # Display the predicted symbol on the GUI with increased text size
                        predicted_label_var.set(f'Predicted: {predicted_symbol}')
                        predicted_label_widget.config(font=("Helvetica", 24))  # Increase text size

                        # Convert the grayscale image to uint8 for display
                        normalized_hand_display = (normalized_hand * 255).astype(np.uint8)

                        # Display the cropped grayscale hand image on the GUI
                        img = Image.fromarray(normalized_hand_display)
                        img = ImageTk.PhotoImage(img)
                        image_label.config(image=img)
                        image_label.image = img

        except Exception as e:
            print("Error:", e)

    # Display the camera image on the GUI
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(img)
    video_label.config(image=img)
    video_label.image = img

    # Call the update_frame function again after 10 ms
    root.after(10, update_frame)

# Create a GUI window
root = tk.Tk()
root.title("Sign Language Recognition")

# Create a label for displaying the video feed
video_label = ttk.Label(root)
video_label.pack()

# Create a label for displaying the predicted symbol with increased text size
predicted_label_var = tk.StringVar()
predicted_label_var.set("Predicted: ")
predicted_label_widget = ttk.Label(root, textvariable=predicted_label_var, font=("Helvetica", 24))
predicted_label_widget.pack()

# Create a label for displaying the cropped hand image
image_label = ttk.Label(root)
image_label.pack()

# Open a connection to the webcam (0 for default webcam)
cap = cv2.VideoCapture(0)

# Start updating the frame in the GUI
update_frame()

# Run the GUI main loop
root.mainloop()

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
