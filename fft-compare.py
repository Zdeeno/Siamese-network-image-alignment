import numpy as np
import scipy
from statistics import mode
from numpy import unravel_index

def handleNegatives(offset, length):
    if offset > length/2:
        offset -= length
    return offset

def fft_1d(a, b):

    x = scipy.fft.fft(a)
    y = scipy.fft.fft(b)
    
    f = lambda x: x.imag
    y = f(y)
    fab = np.multiply(x, y)
    
    res = scipy.fft.ifft(fab)
    res = f(res)
    idx = res.argmax()
    result = handleNegatives(idx, a.shape[0])
    return result

def fft_2d(a, b):
    
    x = scipy.fft.fft(a)
    y = scipy.fft.fft(b)
    
    f = lambda x: x.imag
    y = f(y)
    fab = np.multiply(x, y)
    
    res = scipy.fft.ifft(fab)
    res = f(res)
    idx = res.argmax()
    z = unravel_index(res.argmax(), res.shape)[1]
    result = handleNegatives(z, a.shape[1])
    return result

def fft_3d(a, b):
    if len(a.shape) == 2:
        return fft_2d(a, b)
    a = a.swapaxes(1, 2)
    a = a.swapaxes(0, 1)
    b = b.swapaxes(1, 2)
    b = b.swapaxes(0, 1)
    offsets = []
    for layer in range(len(a)):
        offset = fft_2d(a[layer], b[layer])
        offsets.append(offset)
    m = mode(offsets)
    return m

if __name__ == "__main__":
    
    a = np.array([[6, 3, 4, 1, 4, -6], [1, 5 ,-2, 7, 8, 0]]) #some shitty base

    zero = np.array([[6, 3, 4, 1, 4, -6], [1, 5 ,-2, 7, 8, 0]]) #identical to base
    one = np.array([[3, 4, 1, 4, -6, 8], [5 ,-2, 7, 8, 0, -3]]) #everything shifted one to the left
    two = np.array([[4, 1, 4, -6, 8, 2], [-2, 7, 8, 0, -3, 2]]) # everything shifted two

    print("Should be zero: ", fft_2d(a, zero)) 
    print("Should be one: ", fft_2d(a, one)) 
    print("Should be two: ", fft_2d(a, two)) 

    
