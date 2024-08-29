import cv2
import numpy as np 
from scipy import signal

# Create gray image which consists of unit impulse at the centre
rows = 1024
cols = 1024
img = signal.unit_impulse((rows,cols),'mid')
cv2.imwrite('inputimg.png', img)
cv2.imshow('Input Image',img)

# Compute 2D convolution of image with the kernel
def conv2d(image,kernel,padding):
    
    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))
    # Initialize the output and padding arrays
    output = np.zeros(image.shape)
    rows = image.shape[0] + (kernel.shape[0]-1)
    cols = image.shape[1] + (kernel.shape[1]-1)
    pad = np.zeros((rows,cols))
    
    # Padding
    if padding == 'zero':
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pad[i+ int((kernel.shape[0]-1)/2),j+ int((kernel.shape[1]-1)/2)] = image[i,j]
                
    if padding == 'wraparound':
        pad[0,0] = image[0,0]
        pad[pad.shape[0]-1,0] = image[image.shape[0]-1,0]
        pad[0,pad.shape[1]-1] = image[0,image.shape[1]-1]  
        pad[pad.shape[0]-1,pad.shape[1]-1] = image[image.shape[0]-1,image.shape[1]-1] 
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pad[i+ int((kernel.shape[0]-1)/2),j+ int((kernel.shape[1]-1)/2)] = image[i,j]       
        for k in range(1,pad.shape[0]-1):
            pad[k,0] = image[k-1,image.shape[1]-1]
            pad[k,pad.shape[1]-1] = image[k-1,0]
        for l in range(1,pad.shape[1]-1):
            pad[0,l] = image[image.shape[0]-1,l-1] 
            pad[pad.shape[0]-1,l] = image[0,l-1]
            
    if padding == 'copyedge':
        pad[0,0] = image[0,0]
        pad[pad.shape[0]-1,0] = image[image.shape[0]-1,0]
        pad[0,pad.shape[1]-1] = image[0,image.shape[1]-1]  
        pad[pad.shape[0]-1,pad.shape[1]-1] = image[image.shape[0]-1,image.shape[1]-1] 
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pad[i+ int((kernel.shape[0]-1)/2),j+ int((kernel.shape[1]-1)/2)] = image[i,j]
        for k in range(1,pad.shape[0]-1):
            pad[k,0] = image[k-1,0]
            pad[k,pad.shape[1]-1] = image[k-1,image.shape[1]-1]
        for l in range(1,pad.shape[1]-1):
            pad[0,l] = image[0,l-1]
            pad[pad.shape[0]-1,l] = image[image.shape[0]-1,l-1]
            
    if padding == 'reflectacrossedge':
        pad[0,0] = image[0,0]
        pad[pad.shape[0]-1,0] = image[image.shape[0]-1,0]
        pad[0,pad.shape[1]-1] = image[0,image.shape[1]-1]  
        pad[pad.shape[0]-1,pad.shape[1]-1] = image[image.shape[0]-1,image.shape[1]-1] 
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pad[i+ int((kernel.shape[0]-1)/2),j+ int((kernel.shape[1]-1)/2)] = image[i,j]
        for k in range(1,pad.shape[0]-1):
            pad[k,0] = image[k-1,1]
            pad[k,pad.shape[1]-1] = image[k-1,image.shape[1]-2]
        for l in range(1,pad.shape[1]-1):
            pad[0,l] = image[1,l-1]
            pad[pad.shape[0]-1,l] = image[image.shape[0]-2,l-1]
            
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            output[y, x]=(kernel * pad[y: y+kernel.shape[0], x: x+kernel.shape[1]]).sum()  
    return output,pad

kernel = 1/9*(np.ones([55,55])) # 55*55 matrix with all values equal to 255
output,pad = conv2d(img,kernel,'zero')
cv2.imshow('Output Image', output)
cv2.imwrite("outputimg.png",output)       