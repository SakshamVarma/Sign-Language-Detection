import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
from keras.models import load_model
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

model = load_model('sign_language_model_data_aug_new3.h5')

alphabet_symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def update_frame():
    ret, frame = cap.read()

    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    height, width, _ = frame.shape
                    landmarks = [(int(l.x * width), int(l.y * height)) for l in hand_landmarks.landmark]

                    padding = 30 
                    min_x = max(0, min(landmarks, key=lambda p: p[0])[0] - padding)
                    max_x = min(width, max(landmarks, key=lambda p: p[0])[0] + padding)
                    min_y = max(0, min(landmarks, key=lambda p: p[1])[1] - padding)
                    max_y = min(height, max(landmarks, key=lambda p: p[1])[1] + padding)
                    box_width = max_x - min_x
                    box_height = max_y - min_y
                    max_dim = max(box_width, box_height)

                    new_min_x = max(0, min_x - (max_dim - box_width) // 2)
                    new_max_x = min(width, max_x + (max_dim - box_width) // 2)
                    new_min_y = max(0, min_y - (max_dim - box_height) // 2)
                    new_max_y = min(height, max_y + (max_dim - box_height) // 2)

                    if new_max_x > new_min_x and new_max_y > new_min_y:
                        
                        cropped_hand = frame[new_min_y:new_max_y, new_min_x:new_max_x]

                        gray_hand = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2GRAY)

                        resized_hand = cv2.resize(gray_hand, (28, 28))

                        white_background = np.ones_like(resized_hand) * 255

                        white_background[0:28, 0:28] = resized_hand

                        normalized_hand = white_background / 255.0
                        input_data = normalized_hand.reshape(1, 28, 28, 1)

                        prediction = model.predict(input_data)
                        predicted_label = np.argmax(prediction)
                        predicted_symbol = alphabet_symbols[predicted_label]

                        predicted_label_var.set(f'Predicted: {predicted_symbol}')
                        predicted_label_widget.config(font=("Helvetica", 24)) 

                        normalized_hand_display = (normalized_hand * 255).astype(np.uint8)

                        img = Image.fromarray(normalized_hand_display)
                        img = ImageTk.PhotoImage(img)
                        image_label.config(image=img)
                        image_label.image = img

        except Exception as e:
            print("Error:", e)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(img)
    video_label.config(image=img)
    video_label.image = img

    root.after(10, update_frame)

root = tk.Tk()
root.title("Sign Language Recognition")
video_label = ttk.Label(root)
video_label.pack()

predicted_label_var = tk.StringVar()
predicted_label_var.set("Predicted: ")
predicted_label_widget = ttk.Label(root, textvariable=predicted_label_var, font=("Helvetica", 24))
predicted_label_widget.pack()

image_label = ttk.Label(root)
image_label.pack()
cap = cv2.VideoCapture(0)
update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
