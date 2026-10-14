import cv2
img = cv2.imread("comic1.jpg")
if img is None:
    raise FileNotFoundError("check comic1.jpg")
h,w=img.shape[:2]
M=cv2.getRotationMatrix2D((w/2,h/2),30,1)
rotated = cv2.warpAffine(img,M,(w,h))
crop = img[100:400,200:600]
print("rotated.shape:",rotated.shape)
print("crop.shape:",crop.shape)
cv2.imshow("comic1", crop)
cv2.imshow("comic1", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()