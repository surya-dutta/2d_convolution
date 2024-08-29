import cv2
import numpy as np 

# Read the input image and convert it to grayscale
img = cv2.imread('lena.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('grayimg.png', img)

# Computing 2D convolution of the image with the kernel
def conv2d(image,kernel,padding):
    
    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))
    # Initialize the output and padding arrays
    output = np.zeros(image.shape)
    rows = image.shape[0] + (kernel.shape[0]-1)
    cols = image.shape[1] + (kernel.shape[1]-1)
    pad = np.zeros((rows,cols))
    
    #Padding
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
            
    cv2.imwrite("paddedimg.png",pad)    
    # Compute element-wise multiplication and add the products
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            output[y, x]=(kernel * pad[y: y+kernel.shape[0], x: x+kernel.shape[1]]).sum()         
    return output,pad

kernel = np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])
output,pad = conv2d(img,kernel,'zero')
cv2.imwrite("outputimg.png",output)       