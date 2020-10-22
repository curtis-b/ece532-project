import numpy as np
import matplotlib.pyplot as plt


# loads an idx1-formatted label vector from the MNIST dataset into ndarray
def get_labels(fname):

    f = open(fname, 'rb')

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
    
    
# loads an idx3-formatted image matrix from the MNIST dataset into ndarray
def get_images(fname):

    f = open(fname, 'rb')

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


# creates a sample image using the first 5 images and labels from the MNIST training set
def sample_image():
    
    X = get_images('data/t10k-images-idx3-ubyte')
    y = get_labels('data/t10k-labels-idx1-ubyte')
    
    img = X[1].reshape((28,28))
    
    fig = plt.figure(figsize=(5,1.5), dpi=300)
    axs = fig.subplots(1,5)
    
    for i in range(5):
        
        axs[i].imshow(X[i].reshape(28,28), cmap='gray')
    
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        
        axs[i].set_xlabel('{:0.0f}'.format(y[i,0]))

    plt.tight_layout()
    plt.savefig('sample5x1.png')

sample_image()