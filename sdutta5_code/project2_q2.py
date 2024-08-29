import cv2
import numpy as np
import matplotlib.pyplot as plt

#Read input image as grayscale
inp = cv2.imread('lena.png',0)
inp = inp.astype('float32')

#Perform scaling on the input image 
rows = inp.shape[0]
cols = inp.shape[1]
rmin = np.amin(inp)
rmax = np.amax(inp)
smin = 0
smax = 1
s = np.zeros((rows,cols))
for i in range(rows):
    for j in range(cols):
        s[i,j] = (((smax-smin)/(rmax-rmin))*(inp[i,j] - rmin)) + smin

#Compute 2D DFT
def DFT2D(img):
    fft = np.fft.fft(img)
    fft = np.transpose(fft)
    fft = np.fft.fft(fft)
    fft = np.transpose(fft)
    return fft

dft_img = DFT2D(s)
angle = np.angle(dft_img)
plt.imshow(np.log(1+abs(dft_img)),cmap='Greys')
plt.savefig('dft_img.png')
plt.imshow(angle,cmap='Greys')
plt.savefig('dft_angle.png')

#Compute 2D IDFT
def IDFT2D(img):
    r = dft_img.shape[0]
    c = dft_img.shape[1]
    d = DFT2D(np.conj(dft_img))
    img = np.real(np.conj(d))
    img = img/(r*c)
    return img

idftimg = IDFT2D(dft_img)
x = np.round(s-idftimg)
plt.imshow(idftimg,cmap='Greys')
plt.savefig('idft_img.png')
cv2.imshow('zero_fig.png',x)
cv2.imwrite('zero_fig.png',x)