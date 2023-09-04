import cv2
import os

# Define the directory where you want to save the dataset
dataset_dir = "asl_dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Define the ASL signs you want to capture and create folders for each sign
asl_signs = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
             "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

for sign in asl_signs:
    sign_dir = os.path.join(dataset_dir, sign)
    if not os.path.exists(sign_dir):
        os.makedirs(sign_dir)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set the image dimensions (70x70 pixels)
width, height = 70, 70

# Set the number of images to capture for each sign
num_images_per_sign = 100

# Capture and save images
for sign in asl_signs:
    for i in range(num_images_per_sign):
        ret, frame = cap.read()
        
        # Resize the image to 70x70 pixels
        frame = cv2.resize(frame, (width, height))
        
        # Save the image to the corresponding sign folder
        img_filename = os.path.join(dataset_dir, sign, f"{sign}_{i}.jpg")
        cv2.imwrite(img_filename, frame)
        
        # Display a live preview of the captured image
        cv2.imshow("Captured Image", frame)
        
        # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
