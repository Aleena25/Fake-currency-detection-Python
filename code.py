import cv2
import os
import numpy as np
import pickle

img = cv2.imread(r"C:\Users\SHAFEEK\Anaconda3\Scripts\currency detector\2000\2000front.jpeg")  # reading the image file
img1 = cv2.imread(r"C:\Users\SHAFEEK\Anaconda3\Scripts\currency detector\2000\2000back.jpeg")
res1 = cv2.resize(img, (800, 300))  # resizing the img for display
res2 = cv2.resize(img1, (800, 300))  # resizing the img for display
image = np.concatenate((res1, res2), axis=0)  # concatenating images for display
img_median = cv2.medianBlur(image, 3)  # Add median filter to image
gray = cv2.cvtColor(img_median, cv2.COLOR_BGR2GRAY)  # converting to gray scale
edges = cv2.Canny(gray, 60, 180)  # canny edge detection
th2 = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)              # segmentation using adaptive thresholding

cv2.imshow('original image', image)  # original images
cv2.imshow('noise filtered', img_median)   # filtered images
cv2.imshow('gray scale', gray)  # gray scale
cv2.imshow('edge detected', edges)  # edge detected
cv2.imshow('segmented', th2)  # segmented
cv2.waitKey(0)        # Wait for a key press to
cv2.destroyAllWindows()  # close the img window

# for saving segmented image
#with open('saved.pkl', 'wb') as f:
#    pickle.dump(th2, f)

outfile = open('saved', 'wb')
pickle.dump(th2, outfile)
outfile.close()
