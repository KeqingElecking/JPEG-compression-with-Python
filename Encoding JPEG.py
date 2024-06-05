import cv2 
import numpy as np
import math
Q = np.loadtxt('D:\\Local Disk\\Python\\Quantization_Matrix.txt', delimiter='\t', dtype=int)
pi = math.pi
#This need resize function
def resize(arr, w, h):
    if (h%8!=0):
        h += (8 - h%8)
    if (w%8!=0):
        w += (8 - w%8)
    arr = cv2.resize(arr, (w, h), cv2.INTER_LINEAR)
    return arr, w, h    
def color_detect(arr, y, cb, cr, h, w):
    for i in range(0, h):
        for j in range(0, w):
            color = arr[i][j]
            y[i][j] = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
            cb[i][j] = -0.1687*color[0] + 0.3313*color[1] + 0.5*color[2] + 128
            cr[i][j] = 0.5*color[0] - 0.4187*color[1] - 0.0813*color[2] + 128
    return y, cb, cr
def down_sampling(arr, h, w):
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            avg = (arr[i][j] + arr[i+1][j] + arr[i][j+1] + arr[i+1][j+1])/4
            arr[i][j] = avg
            arr[i+1][j] = avg
            arr[i][j+1] = avg
            arr[i+1][j+1] = avg
def dct_trans(arr, h, w):   
    for i in range(0, h, 8):
        for j in range(0, w, 8):
 
            # ci and cj depends on frequency as well as
            # number of row and columns of specified matrix
            if (i == 0):
                ci = 1 / (8 ** 0.5)
            else:
                ci = (2 / 8) ** 0.5
            if (j == 0):
                cj = 1 / (8 ** 0.5)
            else:
                cj = (2 / 8) ** 0.5
 
            # sum will temporarily store the sum of
            # cosine signals
            sum = 0
            for k in range(0, h, 8):
                for l in range(0, w, 8):
                    dct1 = (arr[k][l] - 128) * math.cos((2 * k + 1) * i * pi / (
                        2 * 8)) * math.cos((2 * l + 1) * j * pi / (2 * 8))
                    sum = sum + dct1
            arr[i][j] = ci * cj * sum
# Quantization Matrix: Q50
def quantization(arr, h, w):  
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            arr[i:(i+8), j:(j+8)] = np.round(arr[i:(i+8), j:(j+8)]/Q)
# FINAL: Zigzag + Huffman encode
img = cv2.imread('D:\\Local Disk\\Python\\sample.bmp') 
dummy = img
x = img.shape[0]
y = img.shape[1]
dummy, y, x = resize(dummy, y, x)
y_img = np.empty(shape=(x, y))
cb_img = np.empty(shape=(x, y))
cr_img = np.empty(shape=(x, y))
y_img, cb_img, cr_img = color_detect(dummy, y_img, cb_img, cr_img, x, y)
print(y_img)
cv2.dct(y_img, y_img, cv2.DCT_INVERSE)
down_sampling(cb_img, x, y)
down_sampling(cr_img, x, y)
cv2.dct(cb_img, cb_img, cv2.DCT_INVERSE)
cv2.dct(cr_img, cr_img, cv2.DCT_INVERSE)
quantization(y_img, x, y)
quantization(cb_img, x, y)
quantization(cr_img, x, y)
# dct_trans(y_img, x, y)
print(y_img)
# print(img.shape)