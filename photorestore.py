# ___________________________PHOTO RESTORATION USING INPAINTING__________________________________________________________-

import numpy as np
import cv2

image = cv2.imread('/home/paython/Desktop/images/abraham.jpg')
cv2.imshow('Original image',image)
cv2.waitKey(0)

marked_damages = cv2.imread('/home/paython/Desktop/images/abraham.jpg',0)
cv2.imshow('Marked Damages',marked_damages)
cv2.waitKey(0)

ret, thresh1 = cv2.threshold(marked_damages,254,255,cv2.THRESH_BINARY)
cv2.imshow('Thresold Binary',thresh1)
cv2.waitKey(0)

kernal = np.ones((7,7),np.uint8)
mask = cv2.dilate(thresh1,kernal,iterations=1)
cv2.imshow('Dilated mask',mask)
cv2.imwrite('/home/paython/Desktop/images/abraham.jpg',mask)

cv2.waitKey(0)
restored = cv2.inpaint(image,mask,3,cv2.INPAINT_TELEA)

cv2.imshow('Restored',restored)
cv2.waitKey(0)
cv2.destroyAllWindows()