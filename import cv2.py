import cv2 
import imutils

img = cv2.imread("sample.png")
resizeImg = imutils.resize(img, width=100)

cv2.write("resizedImage.jpg", resizeImg)