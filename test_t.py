import cv2

image = cv2.imread('/pack/temp/jpg/202107141381.jpg', 1)
cv2.imwrite('/pack/temp/text/1.jpg', image)
