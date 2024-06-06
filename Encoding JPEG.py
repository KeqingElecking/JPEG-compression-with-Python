import cv2 
import numpy as np
import array
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
def dct_trans(arr):
    for i in range(8):
        for j in range(8):
 
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
            for k in range(8):
                for l in range(8):
                    dct1 = (arr[k][l] - 128) * math.cos((2 * k + 1) * i * pi / (
                        2 * 8)) * math.cos((2 * l + 1) * j * pi / (2 * 8))
                    sum = sum + dct1
            arr[i][j] = ci * cj * sum
def dct_full(arr, h ,w):
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = arr[i:(i+8), j:(j+8)]
            dct_trans(block)
# Quantization Matrix: Q50
def quantization(arr, h, w):  
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            arr[i:(i+8), j:(j+8)] = np.round(arr[i:(i+8), j:(j+8)]/Q)
# FINAL: Zigzag + Huffman encode
def zigzag(arr):
    n = 8
    zigzag_order = []
    for i in range(2 * n - 1):
        if i < n:
            if i % 2 == 0:
                x, y = i, 0
                while x >= 0:
                    zigzag_order.append(arr[x][y])
                    x -= 1
                    y += 1
            else:
                x, y = 0, i
                while y >= 0:
                    zigzag_order.append(arr[x][y])
                    x += 1
                    y -= 1
        else:
            if i % 2 == 0:
                x, y = n - 1, i - n + 1
                while y < n:
                    zigzag_order.append(arr[x][y])
                    x -= 1
                    y += 1
            else:
                x, y = i - n + 1, n - 1
                while x < n:
                    zigzag_order.append(arr[x][y])
                    x += 1
                    y -= 1
    return zigzag_order
def zigzagfull(arr, h, w):
    zigzag_full = []
    for i in range (0, h, 8):
        for j in range (0, w, 8):
            zigzag_full.extend(zigzag(arr[i:(i+8), j:(j+8)]))
    return zigzag_full
def run_length_encode(arr):
    rle = []
    cnt0 = 0
    for i in range(1, 64):
        if arr[i] == 0:
            cnt0 += 1
        else:
            while cnt0 > 15:
                rle.append((15, 0))
                cnt0 -= 16
            rle.append((cnt0, int(arr[i])))
            cnt0 = 0
    rle.append((0, 0))  # EOB
    return rle
def rlefull(arr):
    rle_full = []
    for i in range (0, len(arr), 64):
        rle_full.extend(run_length_encode(arr[i:(i+64)]))
    return rle_full

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
dct_full(y_img, x, y)
print(y_img)
# cv2.dct(y_img, y_img, cv2.DCT_INVERSE)
# down_sampling(cb_img, x, y)
# down_sampling(cr_img, x, y)
# dct_full(cb_img, x, y)
# dct_full(cr_img, x, y)
quantization(y_img, x, y)
print(y_img)
# quantization(cb_img, x, y)
# quantization(cr_img, x, y)
y_zigzag = zigzagfull(y_img, x, y)
# print(y_zigzag)
y_rle = rlefull(y_zigzag)
print(y_rle)
# print(img.shape)