"""
Util code needed to convert a dataset into training and testing data of the correct format
Micah Painter, Elle McMahon
Date: 04/04/2024
"""

# imports
from PIL import Image
import numpy as np
from pathlib import Path
import random
import tensorflow as tf

def main():
    # choose path based on who is running the code, mpainter path seems to be public, so use that if in doubt
    #path_name = "/homes/emcmahon/data/ASL_Alphabet_Dataset/asl_alphabet_train"
    path_name = "/homes/mpainter/cs360/project/archive/ASL_Alphabet_Dataset/asl_alphabet_train"
    load_data(path_name, True)


def load_data(path_name, isTrain):
    '''
    Load the data given a path. The path should have a subfolders representing each label. 
    Returns the X and y data as two tensors. 
    '''

    # chose an image size of 32 x 32 to make the runtime shorter, increase if accuracy is too low
    image_size = 32

    # creates a list of all the paths, with each path directing towards a folder of pictures of a particular label
    image_path = Path(path_name)
    image_path_list = list(image_path.glob("*/*.jpg"))
    print(len(image_path_list))

    # the data labels and the numeric representations of them (starting with A=0)
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'] 
    numbers = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

    # initialize variables
    X_train = tf.fill([0, image_size, image_size, 3], 0)
    y_train = tf.fill([0, ], 0)

    # for each letter, find all the pixel values and put them in a tensor
    for i in range(len(letters)):
        X_data, y_data = create_batch(path_name, letters[i], numbers[i], image_size, isTrain)
        X_train = tf.concat([X_train,X_data], 0)
        y_train = tf.concat([y_train, y_data], 0)

    # print the shape of X and y data to make sure it is the correct shape
    print("X shape: ", X_train.shape)
    print("y shape: ", y_train.shape)

    # put X and y data into an array and flatten the y data
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()

    #finding mean and std of training
    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)

    #then subtract off this mean from each dataset and divide by the std
    X_train = (X_train - mean_pixel)/std_pixel

    return X_train, y_train
    

def create_batch(path, letter, number, image_size, isTrain):
    '''
    Given a path to the data, and a speicifc letter or label (and the corresponding value), creates tensors of X and
    y data for that letter. 
    Returns these tensors
    '''
    # Create a list of the paths for a random sample of 1000 images of the given label
    letter_specific_image_path = Path(path+'/'+letter)
    image_path_list = list(letter_specific_image_path.glob("*.jpg"))
    if isTrain == True:
        image_path_list_subset = random.sample(image_path_list, 500) # if it's training data, take a subset
    else:
        image_path_list_subset = random.sample(image_path_list, 5) # use entire testing data

    # resize the images to 32x32, saved as a list
    resized_images = resize_images(image_path_list_subset, image_size)

    # turn the list of images into a tensor of all the pixel RGB values
    tensor_of_images_X = to_RBG(resized_images)

    # create the y data, which is just one number because we are creating tensors by label
    tensor_of_images_y = tf.fill([tensor_of_images_X.shape[0],], number)
    print("X shape:", tensor_of_images_X.shape)
    print("y shape:", tensor_of_images_y.shape)

    # return the tensor of images and a tensor of the labels
    return(tensor_of_images_X, tensor_of_images_y)

def resize_images(path_list, image_size):
    '''
    resizes a given image to 32x32 pixels
    returns that image
    '''
    # initialize list
    resized_images = []
    
    # resize each image to the given size and append to our list
    for one_path in path_list:
        image = Image.open(one_path).resize((image_size,image_size))
        resized_images.append(image)
    
    return resized_images

def to_RBG(image_list):
    '''
    converts a set of images into a nx32x32x3 tensor where n is the number of images, 32x32 is the size of the image, and 3 are RBG values
    this is the format we need for our neural network
    returns this tensor
    '''
    # initialize list
    new_list = []

    # for each image, add the rbg values to a list then convert it to a tensor
    for image in image_list:
        rbg_image = np.array(image)
        new_list.append(rbg_image)
    tensor_of_images = tf.convert_to_tensor(new_list)

    return tensor_of_images


main()
