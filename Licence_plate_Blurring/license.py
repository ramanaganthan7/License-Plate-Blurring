import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure valid file paths for images and cascade
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the paths relative to the base directory
img_path = os.path.join(base_dir, "assests", "car_plate.jpg")
cascade_path = os.path.join(base_dir, "model", "haarcascade_russian_plate_number.xml")


# Load the image
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {img_path}")

# Function to display images using matplotlib
def display(img, title="Image"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)
    ax.set_title(title)
    plt.axis("off")
    plt.show()

# Load the Haar cascade for plate detection
if not os.path.exists(cascade_path):
    raise FileNotFoundError(f"Haar cascade XML file not found at {cascade_path}")
plate_cascade = cv2.CascadeClassifier(cascade_path)

# Function to detect license plates and draw rectangles around them
def detect_plate(img):
    
  
    plate_img = img.copy()
  
    plate_rects = plate_cascade.detectMultiScale(plate_img,scaleFactor=1.3, minNeighbors=3) 
    
    for (x,y,w,h) in plate_rects: 
        cv2.rectangle(plate_img, (x,y), (x+w,y+h), (0,0,255), 4) 
        
    return plate_img
    

# Detect plates and display the result
result = detect_plate(img)
display(result, title="Detected License Plates")

# Function to detect and blur license plates
def detect_and_blur_plate(img):
    
    # fill me in
    plate_img = img.copy()
    roi=img.copy()
  
    plate_rects = plate_cascade.detectMultiScale(plate_img,scaleFactor=1.3, minNeighbors=3)
    
    
    for (x,y,w,h) in plate_rects: 
        roi=roi[y:y+h,x:x+w]
        blur=cv2.medianBlur(roi,7)
        plate_img[y:y+h,x:x+w]=blur
    return plate_img
        
        
    

# Detect and blur plates, then display the result
blurred_result = detect_and_blur_plate(img)
display(blurred_result, title="Blurred License Plates")
