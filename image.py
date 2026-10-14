import cv2
img = cv2.imread("comic1.jpg")
if img is None:
    raise FileNotFoundError("check comic1.jpg")
print("image shape:",img.shape)