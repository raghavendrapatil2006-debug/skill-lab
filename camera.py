import cv2

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Invert pixel values to create a negative photo effect
    inverted_frame = cv2.bitwise_not(frame)

    cv2.imshow("Camera", inverted_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()