
import cv2

# 1. Load the photograph from the current folder
img = cv2.imread("comic1.jpg")

# Check whether the image was loaded successfully
if img is None:
    raise FileNotFoundError("Please keep comic1.jpg in the same folder.")

# 2. Convert the original image from color to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Resize the original color image to 640 x 400 pixels
small = cv2.resize(img, (640, 400))

# Resize the grayscale image to 640 x 400 pixels
small_gray = cv2.resize(gray, (640, 400))

# 4. Get the height and width of the original image
height, width = img.shape[:2]

# 5. Create a rotation matrix to rotate the image by 30 degrees
# The image rotates around its center with a scale factor of 1
M = cv2.getRotationMatrix2D((width//2, height//2), 30, 1)

# 6. Resize the original image to 640 x 400 pixels
normalimg = cv2.resize(img, (640,400))

# 7. Rotate the original image using the rotation matrix
# The output retains the original image dimensions
rotated = cv2.warpAffine(img, M, (width, height))

# 8. Apply Gaussian blur to smooth the image and reduce details
# The kernel size is 255 x 255
blurred = cv2.GaussianBlur(img, (255, 255), -2)

# 9. Resize the rotated image to 640 x 400 pixels
rotated_img = cv2.resize(rotated, (640, 400))

# Resize the blurred image to 640 x 400 pixels
blurredimg = cv2.resize(blurred, (640,400))

# 10. Crop a portion of the original image
# Rows 10 to 399 and columns 20 to 599 are selected
cropimg = img[10:400, 20:600]

# 11. Convert the original image to grayscale again
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 12. Apply binary thresholding using a fixed threshold value of 127
# Pixel values above 127 become white (255)
# Pixel values at or below 127 become black (0)
T, binary = cv2.threshold(gray, 200, 20, cv2.THRESH_BINARY)

# 13. Apply Otsu's thresholding to automatically determine a threshold
# The threshold value is calculated based on the grayscale image
T, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 14. Resize the Otsu thresholded image to 640 x 400 pixels
threshold = cv2.resize(otsu, (640,400))
# threshold = cv2.resize(binary, (640,400))

# 15. Apply canny edge detection
gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur  = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)   # low, high
canny = cv2.resize(edges, (640, 400))


# Display all images in a single window with assigned labels

# Resize the cropped image to fit the layout
crop_resized = cv2.resize(cropimg, (320, 200))

# Convert grayscale and binary images to BGR for display in a common layout
gray_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
threshold_display = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
canny_display = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

# Resize images to the same dimensions
gray_display = cv2.resize(gray_display, (320, 200))
threshold_display = cv2.resize(threshold_display, (320, 200))
canny_display = cv2.resize(canny_display, (320, 200))

# Create a list of images with their assigned labels
images = [
    ("Original", normalimg),
    ("Rotated", rotated_img),
    ("Blurred", blurredimg),
    ("Cropped", crop_resized),
    ("Threshold", threshold_display),
    ("Canny Edges", canny_display)
]

# Add labels above each image
labeled_images = []

for label, image in images:
    image = cv2.resize(image, (320, 200))
    image = image.copy()

    cv2.putText(
        image, label, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (0, 255, 0), 2
    )

    labeled_images.append(image)

# Arrange the images in a grid with 3 columns and 2 rows
row1 = cv2.hconcat(labeled_images[0:3])
row2 = cv2.hconcat(labeled_images[3:6])

final_display = cv2.vconcat([row1, row2])

# Display all images in one window
cv2.imshow("OpenCV Image Processing", final_display)

# Save the grayscale image
cv2.imwrite("GrayImage.jpg", gray)

# Wait until a key is pressed
cv2.waitKey(0)

# Close all OpenCV display windows
cv2.destroyAllWindows()

