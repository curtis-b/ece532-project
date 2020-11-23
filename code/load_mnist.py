"""load_mnist.py

Loads MNIST label and image files and parses into numpy.ndarray objects.
A single image file is represented as a row in the array.

"""


import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


f_training_labels = r'../data/train-labels-idx1-ubyte'
f_training_images = r'../data/train-images-idx3-ubyte'

f_test_labels = r'../data/t10k-labels-idx1-ubyte'
f_test_images = r'../data/t10k-images-idx3-ubyte'


def load_labels(filename):
    """Load the idx1-formatted MNIST label vector into an n x 1 array"""

    f = open(filename, 'rb')

    # FILE FORMAT:
    # bytes 0-1  0x00
    # byte  2    0x08 represents ubyte data
    # byte  3    0x01 represents 1d vector
    # byte  4-7  big-endian representation of vector length

    # clear first 4 bytes
    f.read(4)
    
    # get the vector size
    n = int.from_bytes(f.read(4), byteorder='big')
    
    y = np.zeros((n,1))
    i = 0;
    
    for b in f.read():
        
        y[i,0] = b
        i += 1
        
    f.close()
    
    return y
    
    
def load_images(filename):
    """Load the idx3-formatted MNIST label vector into an n x 784 array"""

    f = open(filename, 'rb')

    # FILE FORMAT:
    # bytes 0-1  0x00
    # byte  2    0x08 represents ubyte data
    # byte  3    0x01 represents 1d vector
    # byte  4-7  big-endian representation of vector length

    # clear first 4 bytes
    f.read(4)
    
    # get the vector size
    n_images = int.from_bytes(f.read(4), byteorder='big')
    n_rows = int.from_bytes(f.read(4), byteorder='big')
    n_cols = int.from_bytes(f.read(4), byteorder='big')
    
    # print('i=', n_images)
    # print('r=', n_rows)
    # print('c=', n_cols)
    
    X = np.zeros(n_images * n_rows * n_cols)
    
    i = 0;
        
    for b in f.read():        
        X[i] = b        
        i += 1
        
    f.close()        
    
    return X.reshape((n_images, n_rows * n_cols))


def sample_image():
    """Load data and generate an array of 5 sample images for display"""    
    
    X = load_images(r'../data/t10k-images-idx3-ubyte')
    y = load_labels(r'../data/t10k-labels-idx1-ubyte')
      
    fig = plt.figure(figsize=(5,1.5), dpi=300)
    axs = fig.subplots(1,5)
    
    for i in range(5):
        
        axs[i].imshow(X[i].reshape(28,28), cmap='gray')
    
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        
        axs[i].set_xlabel('{:0.0f}'.format(y[i,0]))

    plt.tight_layout()
    plt.savefig('sample5x1.png')
    
    
def digit_2_matrix(y):
    """Takes the labels vector, with a numeric value of the digit as the label,
    and return a matrix where j=10 columns each represent a digit and the value
    y'_ij is 1 if y_i=j and -1 if y_i<>j
    """
    y_mat = np.zeros((len(y),10))
    
    for j in range(10):
        y_mat[:,j] = ((y == j)*2-1).flatten()

    return y_mat    
    
    
def matrix_2_digit(y_mat):
    """Reverses the process in digit_2_matrix(), returning a vector of digits 
    from the matrix form. We need to decide what to do if there is more than 1
    digit membership found. For now, we default to zero and select the lowest 
    digit where the data is classified. We could build a second level and train
    specifically between those two values.
    """
    
    y = np.argmax(y_mat, axis=1)

    return y.reshape((y_mat.shape[0],1))