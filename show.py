import cv2
img = cv2.imread("comic1.jpg")
if img is None:
    raise FileNotFoundError("check comic1.jpg")
blurred = cv2.GaussianBlur(img, (15, 15), -10)
cv2.imshow("comic1", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()