import cv2
import numpy as np

def show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Create a white background image (400x600, 3 channels)
img = np.full((400, 600, 3), 255, dtype=np.uint8)

# Draw filled shapes (-1 thickness fills the shape)
cv2.circle(img, (120, 120), 60, (0, 0, 255), -1)        # Red circle
cv2.rectangle(img, (250, 60), (380, 180), (255, 0, 0), -1) # Blue rectangle
cv2.circle(img, (480, 130), 40, (0, 160, 0), -1)       # Green circle
cv2.rectangle(img, (80, 250), (220, 340), (0, 140, 255), -1) # Orange rectangle
cv2.circle(img, (420, 300), 70, (200, 0, 200), -1)     # Purple/Pink circle

show("1 Parts", img)


noise = np.random.randint(0, 100, img.shape[:2])
img[noise < 2] = 0        # black specks
img[noise > 97] = 255     # white specks
show("2 - Noisy camera", img)