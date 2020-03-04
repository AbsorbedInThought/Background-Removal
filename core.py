# -*- coding: utf-8 -*-
"""
Background Removal Tool
Build 1.0
The code below is a simple proof of concept
for the Background Removal Tool project
Created on Tue Feb  4 22:51:35 2020

@author: Muhammad Taha Suhail
"""

#Libraries
##############################################
import glob
import cv2 as cv
import numpy as np
from Classes.crop import Cropper
from Classes.crop_layer import CropLayer
from Classes.edge_recovery import edge_recovery
##############################################

#Path Information
###############################################################
IMAGE_PATH = "test/boxer.jpg"
IMAGES_PATH = "test/*.jpg"
PROTO_PATH = "HED_Model/deploy.prototxt"
TRAINED_MODEL_PATH = "HED_Model/hed_pretrained_bsds.caffemodel"
###############################################################

#Tweaks
#################
COLOR = (0,255,0)
THICKNESS = 3
#################

###############################################################################
#----------------------------------Functions----------------------------------#
###############################################################################

def get_images(IMAGES_DIR):

    cv_img = []
    for img in glob.glob(IMAGES_DIR):
        n = cv.imread(img, 1)
        cv_img.append(n)

    return cv_img


def fill(points, labels): # points = List of tuples

    h, w = im_ff.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    for i in points:
        cv.floodFill(im_ff, mask, (i[0],i[1]), 255)


def draw(event, x, y, flags, param):

    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(im_draw, (x,y), THICKNESS, COLOR)
        h, w = im_ff.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv.floodFill(im_ff, mask, (x,y), 255)

#    elif event == cv.EVENT_MOUSEMOVE:
#            cv.circle(img, (x, y), THICKNESS, COLOR, -1)

#    elif event == cv.EVENT_LBUTTONUP:
#            cv.circle(img, (x, y), THICKNESS, COLOR, -1)


def grab_cut_rect(img, rect):

    mask = np.zeros(img.shape[:2],np.uint8)
    bgd_model = np.zeros((1,65),np.float64)
    fgd_model = np.zeros((1,65),np.float64)
    rec = rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]
    cv.grabCut(img,mask,rec,bgd_model,fgd_model,5,cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]

    return img


def grab_cut_mask(img, mask):

    bgd_model = np.zeros((1,65),np.float64)
    fgd_model = np.zeros((1,65),np.float64)

    mask, bgd_model, fgd_model = cv.grabCut(img, mask, None, bgd_model, fgd_model, 5, cv.GC_INIT_WITH_MASK)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask[:,:,np.newaxis]

    return img


def holistically_nested_edge_detection(image):

    #Performing Edge Detection
    (H, W) = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
    	mean=(104.00698793, 116.66876762, 122.67891434),
    	swapRB=False, crop=False)

    net.setInput(blob)
    hed = net.forward()
    hed = cv.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")
    return hed


def flood_fill(im_in):

    im_th = np.zeros((im_in.shape[0]+2, im_in.shape[1]+2), dtype = np.uint8)
    r,c = im_th.shape
    im_th[1:r-1, 1:c-1] = im_in[:,:]
    im_floodfill = im_th.copy()

    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    cv.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv

    return im_out[1:r-1, 1:c-1], im_floodfill[1:r-1:,1:c-1]


def morphology(image):

    #Thresholding & Gaussian Blur
    #image = cv.GaussianBlur(image,(5,5),0)
    th, im_th = cv.threshold(image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    #Dilation Followed By Erosion, to join broken edges
    kernel = np.ones((5,5),np.uint8)
    dilation = cv.dilate(im_th,kernel,iterations = 1)

    erosion = cv.ximgproc.thinning(dilation)

    kernel = np.ones((3,3),np.uint8)
    dilation = cv.dilate(erosion,kernel,iterations = 1)

    return dilation

def annotation(mask_fg, mask_bg):

    #See np.full()
    mask_gc = np.zeros((im_orig.shape[0], im_orig.shape[1]), dtype = np.uint8)
    mask_gc[mask_gc == 0] = 2 # Unsure Region
    mask_gc[mask_bg == 0] = 0 #Sure BG
    mask_gc[mask_fg == 0] = 1 #Sure FG

    return mask_gc

###############################################################################
#---------------------------------Driver Code---------------------------------#
###############################################################################

#---------------------------------- STEP-1 -----------------------------------#
# Getting Environment Ready

print("---Setting Up Environment---")
#Loading Pre-Trained Model From Disk
print("Loading HED Edge Detector...")
net = cv.dnn.readNetFromCaffe(PROTO_PATH, TRAINED_MODEL_PATH)
cv.dnn_registerLayer("Crop", CropLayer)

print("Reading Image...")
im_orig = cv.imread(IMAGE_PATH, 1) #Reading Image
###############################################################################

#---------------------------------- STEP-2 -----------------------------------#
# Cropping Image

# Cropped contains image with foreground
# and rect has coordinates of bounding box
print("Cropping Image...")
im_cropped, rect = Cropper().process_image(cv.imread(IMAGE_PATH))
###############################################################################

#---------------------------------- STEP-3 -----------------------------------#
# Performing Holistically Nested Edge Detection

im_hed = im_cropped.copy() #To perform HED image
print("Executing Holistically-Nested Edge Detection...")
im_hed = holistically_nested_edge_detection(im_hed)
###############################################################################

#---------------------------------- STEP-4 -----------------------------------#
#Performing Morphological Operations

print("Performing Morphological Operations..")
im_out = morphology(im_hed)
im_morph = im_out.copy()
###############################################################################

#---------------------------------- STEP-5 -----------------------------------#
# Edge Recovery
#im_edge = edge_recovery(im_out)
###############################################################################

#---------------------------------- STEP-6 -----------------------------------#
# Applying GrabCut

print("Applying GrabCut...")
#Paste's computed edge's on sure background
#Sure Background Extraction For GrabCut
sure_bg = np.zeros((im_orig.shape[0], im_orig.shape[1]), dtype = np.uint8)
sure_bg[rect[1]:rect[3], rect[0]:rect[2]] = im_morph[:,:]

im_out, im_ff = flood_fill(sure_bg)

input_gc = annotation(im_ff, im_out)

im_output = im_orig.copy()
cv.imshow("Final", im_output)
im_output = grab_cut_mask(im_output, input_gc)
###############################################################################

#---------------------------------- STEP-7 -----------------------------------#
# Display Results.

print("Displaying Results")
cv.imshow("Floodfill", im_ff)
cv.imshow("Morphed", im_morph)
#cv.imshow("Edge", im_edge)
cv.imshow("Foreground", im_out)
cv.imshow("HED", im_hed)
cv.waitKey(0)
cv.destroyAllWindows()
###############################################################################

#---------------------------------- STEP-8 -----------------------------------#
# User-Interaction

cv.namedWindow('Interaction')
cv.setMouseCallback('Interaction', draw) #Getting Mouse Input

im_draw = im_orig.copy()

while(True):

    cv.imshow('Final', im_output)
    cv.imshow('Interaction', im_draw)

    input_gc = annotation(im_ff, im_out)
    im_output = im_orig.copy()
    im_output = grab_cut_mask(im_output, input_gc)

    cv.imshow('Interaction', im_draw)

    if cv.waitKey():
        pass

    if cv.waitKey(20) & 0xFF == 27:
        break

cv.destroyAllWindows()
###############################################################################
