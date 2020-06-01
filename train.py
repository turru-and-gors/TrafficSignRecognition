"""
turru.and.gors, 2020

This script helps training a Traffic Sign Recognition model
using the German Traffic Sign Recognition Benchmark. See
dataset.py for more details.

To run this script from terminal:
    
    python train.py path/to/GTSRB --width=256 --height=256 --num_classes=43
    
To run this script from an IDE, you can comment line 141 and set path to
your specific path or try parsing the arguments using the menu options.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import dataset
import model

path = "D:/DocsSheila/Databases/GTSRB"
BATCH_SIZE  = 32
BUFFER_SIZE = 100
HEIGHT      = 128
WIDTH       = 128
NUM_CLASSES = 43

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
    global path
    global HEIGHT
    global WIDTH
    global NUM_CLASSES
    
    parser = argparse.ArgumentParser(description = "Training TSR model")
    parser.add_argument("path", type=str, help="Path to dataset")
    parser.add_argument("--height", type=int, help="Image height")
    parser.add_argument("--width", type=int, help="Image width")
    parser.add_argument("--num_classes", type=int, help="Number of classes")
    args = parser.parse_args()
    
    path = args.path
    
    if args.height is not None:
        HEIGHT = args.height
    if args.width is not None:
        WIDTH = args.width
    if args.num_classes is not None:
        NUM_CLASSES = args.num_classes
        
        
def show_sample(dataset, model=None):
    """
    Show five images in the dataset, with their predicted labels as title.
    If no model available, the title is the real label.
    
    :param dataset: Dataset with images to show.
    :type dataset: TensorFlow's Dataset
    
    :param model: Keras model to use for predicting labels.
    :type model: Keras Model
    """
    for img, label in dataset.take(1):
        
        if model is not None:
            pred = model.predict(img)
        
        for index in range(5):
            image = img[index].numpy()
            lbl = label[index].numpy()
            lbl = np.argmax(lbl)
            print("Label: ", lbl)
            
            if model is not None:
                pred_lbl = pred[index]
                pred_lbl = np.argmax(pred_lbl)
                print("Predicted: ", pred_lbl)
            
            #image = (image + 1.0)/2.0
            plt.figure()
            plt.imshow(image)
            plt.show()
            if model is None:
                plt.title(class_names[lbl])
            else:
                plt.title(class_names[pred_lbl])
            
    

if __name__ == "__main__":    
    read_arguments()
    
    train_csv = "Train.csv"
    test_csv = "Test.csv" 
    
    # ===== Datasets
    dataset.IMG_HEIGHT  = HEIGHT
    dataset.IMG_WIDTH   = WIDTH
    dataset.NUM_CLASSES = NUM_CLASSES
    
    # Create Train set -- 39209 samples
    train_ds, ntrain_samples = dataset.create_dataset(path, train_csv, for_training=True)
    train_ds = train_ds.shuffle(buffer_size=1000)
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    # Create Test set -- 12630 samples
    test_ds, ntest_samples = dataset.create_dataset(path, test_csv, for_training=False)
    test_ds = test_ds.repeat()
    test_ds = test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
    # ===== MODEL
    net = model.create_model( (HEIGHT, WIDTH, 3), NUM_CLASSES )
    net.summary()
    
    net.compile(optimizer='adam',
                loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    # ===== TEST MODEL BEFORE TRAINING
    #print("Before training...")
    #show_sample(train_ds, net)
    
    # ===== TRAIN MODEL
    print("\n\nTraining...")
    history = net.fit(train_ds, 
                      validation_data   = test_ds,
                      epochs            = 200,
                      steps_per_epoch   = int(ntrain_samples/BATCH_SIZE),
                      validation_steps  = int(ntest_samples/BATCH_SIZE),
                      verbose           = 1)
    net.save('model.h5')
    print("model.h5 saved")
    
    # Plot history to detect possible overfitting
    plt.figure()
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    
    #test_loss, test_acc = net.evaluate(test_ds, verbose=2, steps=ntest_samples)
    #print(test_loss, test_acc)
    
    # ===== TEST MODEL AFTER TRAINING
    print("\n\nAfter training")
    show_sample(test_ds, net)
        

        
        