# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 22:51:35 2020

@author: Taha
"""

#Libraries
#################################
import cv2 as cv
import numpy as np
from sklearn.svm import SVC
from crop import Cropper
from crop_layer import CropLayer
#################################

#Path Names
IMAGE_PATH = "boxer.jpg"
PROTO_PATH = "deploy.prototxt"
TRAINED_MODEL_PATH = "hed_pretrained_bsds.caffemodel"

###############################################################################
#----------------------------------Functions----------------------------------#
###############################################################################

def grabCut(img, rect):
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    return img

def holistically_nested_edge_detection(image):
    
    #Loading Pre-Trained Model From Disk
    print("Loading HED Edge Detector")
    net = cv.dnn.readNetFromCaffe(PROTO_PATH, TRAINED_MODEL_PATH)
    cv.dnn_registerLayer("Crop", CropLayer)
    
    #Performing Edge Detection
    (H, W) = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
    	mean=(104.00698793, 116.66876762, 122.67891434),
    	swapRB=False, crop=False)
    
    print("Executing Holistically-Nested Edge Detection")
    net.setInput(blob)
    hed = net.forward()
    hed = cv.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")
    return hed

def morphology(image):
    
    #Thresholding
    th, im_th = cv.threshold(image, 100, 255, cv.THRESH_BINARY)
    
    #Dilation Followed By Erosion, to join broken edges
    kernel = np.ones((5,5),np.uint8)
    dilation = cv.dilate(im_th,kernel,iterations = 1)
    erosion = cv.erode(dilation,kernel,iterations = 1)
    th, im_th = cv.threshold(erosion, 150, 255, cv.THRESH_BINARY)
    
    im_floodfill = im_th.copy()
    
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
     
    cv.floodFill(im_floodfill, mask, (0,0), 255);
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv
     
    # Display images.
    cv.imshow("Thresholded Image", im_th)
    cv.imshow("Floodfilled Image", im_floodfill)
    cv.imshow("Foreground", im_out)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    return im_out

###############################################################################
#---------------------------------Driver Code---------------------------------#
###############################################################################

orig_image = cv.imread(IMAGE_PATH) #Reading Image

#Cropped contains image with foreground 
# and rect has coordinates of bounding box
cropped, rect = Cropper().process_image(cv.imread(IMAGE_PATH))
grab_cut = grabCut(orig_image, rect)

#Perfoming HED on GrabCut's Output
#image = grab_cut[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
image = cropped.copy() #To perform on image without GrabCut
hed = holistically_nested_edge_detection(image)

# Holistically-Nested Edge Detection & GrabCut Results
cv.imshow('GrabCut', grab_cut)
cv.imshow("HED", hed)
cv.waitKey(0)
cv.destroyAllWindows()


im_out = morphology(hed)

rows, cols = im_out.shape
train_root = []
labels_root = []

for i in range(0, im_out.shape[0]):
    for j in range(0, im_out.shape[1]):
        if(im_out[i,j] == 0):
            x = cropped[i,j,0],cropped[i,j,1],cropped[i,j,2]
            train_root.append(x)
            labels_root.append(1)
            cropped[i,j] = 0
        else:
            x = cropped[i,j,0],cropped[i,j,1],cropped[i,j,2]
            train_root.append(x)
            labels_root.append(0)
            
cv.imshow("output", cropped)
cv.waitKey(0)
cv.destroyAllWindows()


svm_root = SVC(kernel='linear')
svm_root.fit(train_root, labels_root)

print("Training SVM")
for i in range(0, im_out.shape[0]):
    for j in range(0, im_out.shape[1]):
        prediction = svm_root.predict(orig_image[i,j])
        if(prediction == 0):
            orig_image[i,j] = 0
            
cv.imshow('predict', orig_image)
cv.waitKey(0)
cv.destroyAllWindows()


'''
im = hed.copy()
ret, thresh = cv.threshold(im, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
out = np.zeros((im.shape[0], im.shape[1]), dtype = np.uint8)


for i in range(0, len(contours)):
    cv.drawContours(out, contours, i, (255,255,255), 3)
    cv.imshow('out', out)
    cv.waitKey(0)
cv.destroyAllWindows()
'''

