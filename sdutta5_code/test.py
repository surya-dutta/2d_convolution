import cv2
import numpy as np
  
# Reading the image
image = cv2.imread('lena.png',0) 

# Creating the kernel(2d convolution matrix)
# Box Filter
kernel1 = np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]) 
# Simple First Order Derivative Filter
kernel2 = np.array([[-1,1]]) 
kernel3 = np.array([[-1],[1]])
# Prewitt Filter
kernel4 = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]) #Mx
kernel5 = np.array([[1,1,1],[0,0,0],[-1,-1,-1]]) #My
# Sobel Filter
kernel6 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) #Mx
kernel7 = np.array([[1,2,1],[0,0,0],[-1,-2,-1]]) #My
# Roberts Filter
kernel8 = np.array([[0,1],[-1,0]]) #Mx
kernel9 = np.array([[1,0],[0,-1]]) #My

kernel2 = np.flipud(np.fliplr(kernel2))
# Applying the filter2D() function
img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)
  
# Showing the original and output image
cv2.imshow('Original', image)
cv2.imshow('Convolved', img)  
cv2.waitKey()
cv2.destroyAllWindows()