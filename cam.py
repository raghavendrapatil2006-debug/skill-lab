import cv2
ap=cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    found= detect(frame, 0.5)
    draw(frame, found)
    cv2.imshow("smart camera",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    cap.release()
    cv2.destroyAllWindows()