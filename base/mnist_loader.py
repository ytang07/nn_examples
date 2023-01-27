import pickle
import gzip

import numpy as np

def load_data():
    """loads data from mnist_pkl.gz
    the data is mnist labeled images in the form of tuples
    each tuple contains a first entry containing the image representations
    as numpy ndarrays with 784 values equal to the 28 by 28 pixel
    size of each image and a second entry containing the correct
    label for each corresponding image in the first entry (0 ... 9)
    
    Returns:
        tuple: training, validation, and test data"""
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return (training_data, validation_data, test_data)

def one_hot_encode(y):
    """one hot encodes the y variable into a vector
    Returns:
        np array of size 10: encoded"""
    encoded = np.zeros((10, 1))
    encoded[y] = 1.0
    return encoded

def load_data_together():
    """basically load_data except instead of the datasets being
    represented as tuples, they are represented as lists containing
    tuples where each tuple in the list corresponds to an image
    and it's correct label
    
    Returns:
        tuple: training, validation, and test data """
    train, validate, test = load_data()
    train_x = [np.reshape(x, (784, 1)) for x in train[0]]
    train_y = [one_hot_encode(y) for y in train[1]]
    training_data = zip(train_x, train_y)
    validate_x = [np.reshape(x, (784, 1)) for x in validate[0]]
    validate_y = [one_hot_encode(y) for y in validate[1]]
    validation_data = zip(validate_x, validate_y)
    test_x = [np.reshape(x, (784, 1)) for x in test[0]]
    test_y = [one_hot_encode(y) for y in test[1]]
    testing_data = zip(test_x, test_y)
    return (training_data, validation_data, testing_data)
    