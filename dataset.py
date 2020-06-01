"""
turru.and.gors, 2020

    Convert dataset from PPM to PNG file format.
    
    German Traffic Sign Recognition Benchmark is a large multi-category 
    classification benchmark.
    
    http://benchmark.ini.rub.de/
    
    You can download the PNG version from Kaggle.
    
    https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign#
    
    The dataset has the following tree:
        
        - Train (folder): contains 43 folders numbered 0 to 42. The folder number
                          represent the class number. Inside each folder, there is
                          a set of images in PNG format with names CCCCC_AAAAA_BBBBB.png.
                          C is the class number, A is the frame it was taken from and
                          B is the number of image.
        - Test (folder):  contains a set of images for testing. To know which class they
                          belong to, you need to read the ground truth file.
        - Meta (folder):  Contains drawing images of each class in the dataset.
        - Train.csv:      CSV file with each and every image in the train set. Contains
                          width and height in pixels, position inside the scene where
                          the traffic sign was cut from, class id and path to image.
        - Test.csv:       CSV file with each and every image in the test set. Contains
                          the same information as Train.csv.
        - Meta.csv:       CSV file with each and every image in the meta set. Contains
                          path to image, class id, shape id, color id and sign id.
    
"""

import os
import sklearn.utils as utils
import pandas as pd
import tensorflow as tf

IMG_HEIGHT  = 256
IMG_WIDTH   = 256
NUM_CLASSES = 43

def create_dataset(path, csv_file, for_training=False):
    """
    Create a TensorFlow Dataset from the data in csv_file.
    
    :param path: Path to dataset folder.
    :type path: string
    
    :param csv_file: Name of csv file containing image file and corresponding class number.
    :type csv_file: string
    
    :param batch_size: Batch size of dataset.
    :type batch_size: integer
    
    :return: New dataset containing images and corresponding classes.
    :rtype: TensorFlow Dataset
    """
    # Read csv file
    data = pd.read_csv( os.path.join(path, csv_file) )
    
    # Drop unused columns
    data = data.drop(columns = ["Roi.X1", "Roi.Y1", 
                                          "Roi.X2", "Roi.Y2", 
                                          "Width", "Height"])
    length = data.shape[0]
    
    # Update path
    for index, elem in enumerate(data["Path"]):
        data["Path"][index] = os.path.join(path, elem)
    
    dataset = tf.data.Dataset.from_tensor_slices( (data["Path"].values, data["ClassId"].values) )
    
    if for_training:
        dataset = dataset.map(map_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(map_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
    return dataset, length
        
        
def load_image(image_file, normalize=True):
    """
    Load an image in PNG format and  normalize if requested.
    
    :param image_file: path to image to be loaded.
    :param image_file: string
    
    :param normalize: True if you wish to normalize the image, False otherwise.
    :type normalize: boolean
    
    :return: Loaded image.
    :rtype: TensorFlow Tensor
    """
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image)
    image = tf.cast(image, tf.float32)
    if normalize:
        image = normalize_image(image)
    return image


def resize_image(image, height, width):
    """
    Resize image to desired height and width.
    
    :param image: Image to be resized.
    :type image: TensorFlow Tensor.
    
    :param height: Desired height in pixels.
    :type height: integer
    
    :param width: Desired width in pixels.
    :type width: integer
    
    :return: Resized image.
    :rtype: TensorFlow Tensor
    """
    image = tf.image.resize(image, [height, width])
    return image
        

def normalize_image(image):
    """
    Normalize image to range [-1, 1].
    
    :param image: Image to be normalized.
    :type image: TensorFlow Tensor.
    
    :return: Normalized image.
    :rtype: TensorFlow Tensor
    """
    image = (image / 255.0) 
    return image


def random_crop(image):
    """
    Crop an image to the desired size.
    
    :param image: Image to be cropped.
    :type image: TensorFlow Tensor.
    
    :return: Cropped image.
    :rtype: TensorFlow Tensor
    """
    global IMG_HEIGHT
    global IMG_WIDTH
    
    cropped_image = tf.image.random_crop(image, size=[IMG_WIDTH, IMG_HEIGHT, 3])
    return cropped_image


def random_jitter(image):
    """
    Creates a simple variation of image by scaling up 15% and then randomly
    cropping to the desired size again.
    """
    global IMG_HEIGHT
    global IMG_WIDTH
    
    h = int(IMG_HEIGHT * 1.15)
    w = int(IMG_WIDTH * 1.15)
    img = tf.image.resize(image, [h, w])
    image = random_crop(img)
    return image


def map_image_train(image_path, image_class):
    """
    Maps from (string, integer) to (Tensor, Tensor) for training set. Apply random 
    jitter.
    
    :param image_path: Path to image to be loaded and preprocessed.
    :type image_path: string
    
    :param image_class: Class number for the corresponding image.
    :type image_class: integer
    
    :return: Image, Class in one-hot-encoding
    :rtype: (TensorFlow Tensor, TensorFlow Tensor)
    """    
    # Mapping image
    image = load_image(image_path, normalize=False)
    image = resize_image(image, IMG_HEIGHT, IMG_WIDTH)
    image = normalize_image(image)    
    
    # Augment dataset
    image = random_jitter(image)        
    
    # Mapping class - there are 43 classes
    im_class = tf.one_hot(image_class, depth=NUM_CLASSES)
    return image, im_class


def map_image_test(image_path, image_class):
    """
    Maps from (string, integer) to (Tensor, Tensor) for test set. Does not apply jitter.
    
    :param image_path: Path to image to be loaded and preprocessed.
    :type image_path: string
    
    :param image_class: Class number for the corresponding image.
    :type image_class: integer
    
    :return: Image, Class in one-hot-encoding
    :rtype: (TensorFlow Tensor, TensorFlow Tensor)
    """    
    # Mapping image
    image = load_image(image_path, normalize=False)
    image = resize_image(image, IMG_HEIGHT, IMG_WIDTH)
    image = normalize_image(image)         
    
    # Mapping class - there are 43 classes
    im_class = tf.one_hot(image_class, depth=NUM_CLASSES)
    return image, im_class
    