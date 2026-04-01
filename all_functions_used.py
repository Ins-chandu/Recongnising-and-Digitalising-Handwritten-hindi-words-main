import cv2
import numpy as np
import math
import os

# Desc: first read image, second coverts bgr image to grayscale image
# Finally apply gaussian blur using (5,5) kernel size
# i/p parameters: 
# path: path at which image is located
# returns blurred gray scale image
def load_image(path):
    '''This function prepares an image for further processing by converting it 
    to grayscale and smoothing it, which can help reduce noise and detail.'''

    test_image = cv2.imread(path)
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray, (5, 5), 0) 
    return gray_img

# Desc: resizes the image to scale_percent of its original image (here 50%)
# i/p parameters: 
# gray_img: i/p image
# returns the resized image
def makesmall(gray_img):
    scale_percent = 50
    width = int(gray_img.shape[1] * scale_percent / 100)
    height = int(gray_img.shape[0] * scale_percent / 100)
    gray_img = cv2.resize(gray_img, (width, height))
    return gray_img

# Desc: First make copy of image, second reduce image size using makesmall helper
# cv2.adaptiveThreshold: This function applies adaptive thresholding to the image
# finally return a threshold image in which is binary image with intensities 0 ,255
# foreground pixel has 255 remaining pixels have 0
# i/p parameters: 
# img: which is a gray scale and smoothed image
# returns the threshold image
def remove_noise_and_preprocess(img):
    '''This preprocessing step can be useful for tasks like image segmentation, 
    feature extraction, or preparing an image for OCR (optical character recognition).'''

    gray_img = img.copy() 
    gray_img = makesmall(gray_img)
    threshold = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 199, 20)
    return threshold

# Desc: first reduce the image size using makesmall helper
# Apply binary inverse threshold using otsu method 
# Next apply morphological opening using rectangular kernel size of (1,1) which helps to remove small noises
# Next is contour detection and remove noises
# i/p parameters:
# image: which is a gray scale and smoothed image
# returns a processed binary image
def preprocess(image):
    '''The function returns the processed binary image, which has been resized, thresholded, 
        had noise reduced, and had small, irrelevant contours removed.'''
    
    image = makesmall(image)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        area = cv2.contourArea(c)
        # if area of contour is less than 50 pixels then fill the area with black
        if area < 50:
            cv2.drawContours(opening, [c], -1, 0, -1)
    return opening

# Desc: Normal dist function calculates distance between 2 coordinates (x1,y1) and (x2,y2)
def getdist(x1, x2, y1, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Desc: First use canny edge detection algorithm to find edges in the image. aperturesize is size of sobel kernel
# Second apply hough line transform to detect lines in the edge-detected image from first step
# Third: For each line detected it extracts the distance r from the origin and the angle theta
# And calculate each line end point
# i/p parameters:
# preprocessed_img: preprocessed image after applying otsu etc..
# returns the coordinates of longest detected line (px1, py1) to (px2, py2)
def houghtransform(preprocessed_img):
    '''The houghtransform function detects the longest line in a given 
        grayscale image using the Hough Line Transform. '''

    edges = cv2.Canny(preprocessed_img, 50, 150, apertureSize=3)
    canny_img_path = os.path.join("./segmented_characters/", f'canny_img.png')
    cv2.imwrite(canny_img_path, edges)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, preprocessed_img.shape[1] // 10)

    px1, px2, py1, py2 = -1, -1, -1, -1 # initialize
    if lines is None:
        return px1, px2, py1, py2 # if no line is detected in image return 
    
    mxd = 0 # current max line length
    for r, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # Calculate current line length if > mxd update mxd and pxi values
        curd = getdist(x1, x2, y1, y2)
        if curd > mxd:
            mxd = curd
            px1, px2, py1, py2 = x1, x2, y1, y2

    return px1, px2, py1, py2

# Desc: First make a copy of the image and store it
# Second remove horizontal lines
# Third Apply dilation to the image using 5x5 kernel
# Fourth find the contours in the image
# i/p parameters:
# img: rotated horizontal image
# Note:::: It just returns the cropped version in which word exists. It returns cropped area of complete
# word it doesnot returns a header line removed word
def word_segmentation(img):
    '''This function is useful for extracting the most prominent feature (e.g., a word or text region) 
        from a binary image. It removes horizontal lines, connects nearby regions, and then finds and crops
        the largest contour, making it suitable for text extraction or word recognition tasks.'''

    cpyimg = img.copy()

    # Remove horizontal lines, Iterates through each row of the image to count the number
    # of white pixels. If more than 85% of the pixels in a row are white, it sets the entire 
    # row to black (0).
    for i in range(img.shape[0]):
        cnt = 0
        for j in range(img.shape[1]):
            if img[i][j] == 255:
                cnt += 1
        percent = (100.0 * cnt) / img.shape[1]
        if percent > 85:
            for j in range(img.shape[1]):
                img[i][j] = 0

    kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=4)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = list(contours)
    contours.sort(reverse=True, key=cv2.contourArea)
    contour = contours[0] # larget contour from above sorted list of contours
    x, y, w, h = cv2.boundingRect(contour)
    cropped = cpyimg[y:y + h, x:x + w]
    return cropped

def extractroi(img):
    cpyimg = img.copy()
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=2)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = list(contours)
    contours.sort(reverse=True, key=cv2.contourArea)
    contour = contours[0]
    x, y, w, h = cv2.boundingRect(contour)
    cropped = cpyimg[y:y + h, x:x + w]
    return cropped

def check(img):
    cnt = 0
    for i in range(img.shape[0]//3, img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] == 255:
                cnt += 1
    return cnt > 10

def predictchar(image,model):
	image=cv2.resize(image,(64,64))
	image=image*1/255.0
	image = np.expand_dims(image, axis=0)
	image = np.expand_dims(image, axis=3)
	lists = model.predict(image)[0]
	return np.argmax(lists)
