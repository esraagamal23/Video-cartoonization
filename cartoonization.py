# importing libraries
import cv2
import math
import numpy as np
from numba import njit

@njit
def gaussian(x, sigma):
    return (1.0 / math.sqrt(2 * math.pi) * sigma) * math.exp(-1 * (x ** 2) / (2 * sigma ** 2))

# get the histogram of the image
@njit
def get_hist(img):
    hist_img = np.zeros((256))
    for i in range(len(img)):
        for j in range(len(img[0])):
            hist_img[img[i][j]] +=1
    return hist_img

@njit
def find_median(histogram, aSize):
    histogram = np.cumsum(histogram)
    for i in range(len(histogram)):
        if histogram[i] > (aSize ** 2) / 2:
            return i
    return 0

@njit
def add_to_hist(histogram, arr):
    for i in arr:
        histogram[i] += 1

@njit
def sub_from_hist(histogram, arr):
    for i in arr:
        if(histogram[i]):
            histogram[i] -= 1

@njit
def bilateral_filter_own(img, aSize):
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
    # get the gray image for easy calculations
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
    img = cv2.imread("photo2.jpg")
    cartoon = cartoonize(img, 7)
    cv2.imshow("Image", cartoon)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
