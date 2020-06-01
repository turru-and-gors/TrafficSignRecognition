"""
turru.and.gors , 2020

This script tests the TSR trained model.

To run this script from terminal:
    
    python test.py model_name.h5 path/to/test/images --height=128 --width=128
    
To run this script from an IDE, you can comment line 97 and set test_path and
model_path to your specific needs or try parsing the arguments using the menu options.
"""

import argparse
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

import dataset

model_path  = ''
test_path   = ''
HEIGHT      = 128
WIDTH       = 128

class_names = ['Speed limit (20km/h)',       
          'Speed limit (30km/h)',         
          'Speed limit (50km/h)',         
          'Speed limit (60km/h)', 
          'Speed limit (70km/h)',       
          'Speed limit (80km/h)',         
          'End of speed limit (80km/h)',       
          'Speed limit (100km/h)',
          'Speed limit (120km/h)',      
          'No passing',      
          'No passing for vechiles over 3.5 metric tonsk', 
          'Right-of-way at the next intersection',
          'Priority road',
          'Yield',
          'Stop',
          'No vechiles',
          'Vechiles over 3.5 metric tons prohibited',
          'No entry',
          'General caution',
          'Dangerous curve to the left',
          'Dangerous curve to the right',
          'Double curve',
          'Bumpy road',
          'Slippery road',
          'Road narrows on the right',
          'Road work',
          'Traffic signals',
          'Pedestrians',
          'Children crossing',
          'Bicycles crossing',
          'Beware of ice/snow',
          'Wild animals crossing',
          'End of all speed and passing limits',
          'Turn right ahead',
          'Turn left ahead',
          'Ahead only',
          'Go straight or right',
          'Go straight or left',
          'Keep right',
          'Keep left',
          'Roundabout mandatory',
          'End of no passing',
          'End of no passing by vechiles over 3.5 metric tons']


def read_arguments():
    """
    Argument parser - to invoke this script directly from terminal.
    """
    global model_path
    global test_path
    global HEIGHT
    global WIDTH
    
    parser = argparse.ArgumentParser(description = "Testing TSR model")
    parser.add_argument("model_path", type=str, help="Path to dataset")
    parser.add_argument("test_path", type=str, help="Path to test images")
    parser.add_argument("--height", type=int, help="Image height")
    parser.add_argument("--width", type=int, help="Image width")
    args = parser.parse_args()
    
    model_path  = args.model_path
    test_path   = args.test_path
    if args.height is not None:
        HEIGHT      = args.height
    if args.height is not None:
        WIDTH       = args.width


if __name__ == "__main__":
    read_arguments()
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    img_paths = os.listdir(test_path)
    index = 0
    for file in img_paths:
        if ".png" not in file:
            continue
        
        # Load and preprocess image
        file = os.path.join(test_path, file)
        image = cv2.imread(file)
        image = cv2.resize(image, (WIDTH, HEIGHT))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image / 127.5) - 1
        
        # Predict label for current image
        im_tensor = tf.convert_to_tensor(image)
        im_tensor = tf.expand_dims(im_tensor, 0)
        pred = model.predict(im_tensor)
        pred_label = tf.math.argmax(pred, axis=1)
        pred_label = pred_label.numpy()
        pred_label = pred_label[0]
        
        # Display results
        image = (image + 1.0)/2.0 
        plt.figure()
        plt.imshow(image)
        plt.title(class_names[pred_label])
        print(pred_label)
        
        index += 1
        if index > 10:
            break