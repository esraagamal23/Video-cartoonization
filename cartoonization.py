# importing libraries
import cv2
import math
import numpy as np
from numba import njit

@njit
def gaussian(x, sigma):
    '''
    Gaussian function is resposible for calculating the corresponding gaussian kernal value

    Key arguments:
    x : Value to be evaluated for
    sigma : The standard diviation
    '''
    return (1.0 / np.sqrt(2 * np.pi) * sigma) * np.exp(-1 * (x ** 2) / (2 * sigma ** 2))

@njit
def get_hist(img):
    '''
    Get the histogram of the given image

    key arguments:
    img: The needed image
    '''
    hist_img = np.zeros((256))
    for i in range(len(img)):
        for j in range(len(img[0])):
            hist_img[img[i][j]] +=1
    return hist_img

@njit
def find_median(histogram, aSize):
    '''
    Get the median of the given window from its histogram
    
    Details: If we have a histogram so it's ordered by default, then if we sum the values from the one end
    till we reached the value half the size of the window so, the corresponding pixel is the median value

    key arguments:
    img: The needed image
    '''
    histogram = np.cumsum(histogram)
    for i in range(len(histogram)):
        if histogram[i] > (aSize ** 2) / 2:
            return i
    return 0

@njit
def add_to_hist(histogram, arr):
    '''
    This function add the next vertical raw beside the window to the histogram

    key arguments:
    histogram: The histogram
    arr: The array to be added
    '''
    for i in arr:
        histogram[i] += 1

@njit
def sub_from_hist(histogram, arr):
    '''
    This function subtract the previous vertical raw beside the window to the histogram

    key arguments:
    histogram: The histogram
    arr: The array to be subtract
    '''
    for i in arr:
        if(histogram[i]):
            histogram[i] -= 1

@njit
def bilateral_filter_own(img, aSize):
    '''
    This function apply the median filter with Huang's algorithm to blur the image and preserve the edges

    key arguments:
    img: The needed image
    aSize: The window diameter and must be odd
    '''
    source_img = np.copy(img)
    raw, col = img.shape
    offset = aSize // 2
    i = offset
    j = offset
    while i in range(offset, raw - offset):
        hist_img = get_hist(img[i - offset: i + offset, 0 : aSize])
        source_img[i][j] = find_median(hist_img, aSize)
        while j in range(offset + 1, col - offset):
            add_to_hist(hist_img, img[i - offset: i + offset + 1, j + offset])
            sub_from_hist(hist_img, img[i - offset: i + offset + 1, j - offset])
            source_img[i][j] = find_median(hist_img, aSize)
            j += 1
        i += 1
        j = offset + 1
    return source_img

def cartoonize(img, aSize):
    '''
    This function cartoonize the image by applying multiple filter
    filters:
        - median filter -> to blur the image preserving its edges
        - edges with adaptive threshold -> to get the edges of the image
        - bitwise and -> to merge the blured image with the edges
    
    key arguments:
    img: The needed image
    aSize: The window size of the blur filter
    '''
    # get the gray image for easy calculations
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply median filter
    for i in range(3):
        img[:, :, i] = bilateral_filter_own(img[:, :, i], aSize)
    # get the edges
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    # bitwise two images
    cartoon = cv2.bitwise_and(img, img, mask=edges)
    return cartoon

if __name__ == "__main__":
    
    # reading image
    img = cv2.imread("Images/giraffe.jpg")
    cartoon = cartoonize(img, 7)
    cv2.imshow("Cartoon", cartoon)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
