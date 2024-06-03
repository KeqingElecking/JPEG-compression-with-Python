
import cv2 
import numpy as np
def color_detect(arr, y, cb, cr, h, w):
    for i in range(0, h):
        for j in range(0, w):
            color = arr[i][j]
            y[i][j] = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
            cb[i][j] = -0.1687*color[0] + 0.3313*color[1] + 0.5*color[2] + 128
            cr[i][j] = 0.5*color[0] - 0.4187*color[1] - 0.0813*color[2] + 128
    return y, cb, cr
def down_sampling(arr):
    
img = cv2.imread('D:\Local Disk\Python\sample.bmp') 
x = img.shape[0]
y = img.shape[1]
y_img = np.empty(shape=(x, y))
cb_img = np.empty(shape=(x, y))
cr_img = np.empty(shape=(x, y))
y_img, cb_img, cr_img = color_detect(img, y_img, cb_img, cr_img, x, y)
print(y_img)
# print(img.shape)