
import sys
import cv2
import numpy as np


def make_test_image():
    """Noisy white image with 5 coloured shapes."""
    img = np.full((1500, 700, 3), 245, np.uint8)
    cv2.circle(img, (120, 120), 60, (90, 90, 250), -1)                    # circle
    cv2.rectangle(img, (260, 60), (420, 180), (60, 120, 230), -1)         # box (160x120)
    cv2.rectangle(img, (500, 70), (620, 190), (200, 120, 80), -1)         # square (120x120)
    tri = np.array([[120, 420], [60, 320], [180, 320]])
    cv2.fillPoly(img, [tri], (80, 180, 90))                               # triangle
    cv2.circle(img, (400, 380), 45, (160, 90, 200), -1)                   # circle
    noise = np.random.randint(0, 60, img.shape[:2], np.uint8)
    mask = np.random.rand(*img.shape[:2]) < 0.03                          # salt & pepper
    img[mask] = (noise[mask][:, None])
    img = cv2.add(img, np.random.randint(0, 25, img.shape, np.uint8))
    return img


def classify(contour):
    """Name the shape from its outline."""
    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * area / (peri * peri) if peri else 0

    # Check roundness first so a circle is never mistaken for a 4-corner shape.
    if circularity > 0.85:
        return "circle"

    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    corners = len(approx)

    if corners == 3:
        return "triangle"
    if corners == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(h)
        return "square" if 0.9 <= ratio <= 1.1 else "box"
    return f"shape({corners})"


def process(img, blur_k, thresh_val, min_area, kernel_k):
    # 1. grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. denoise (kernel must be odd)
    blur_k = max(1, blur_k) | 1
    blurred = cv2.medianBlur(gray, blur_k)

    # 3. threshold (inverted: objects become white on black)
    #    thresh_val == 0 -> automatic Otsu; otherwise use the slider value.
    if thresh_val == 0:
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # 4. morphology: remove specks, fill holes
    k = max(1, kernel_k)
    kernel = np.ones((k, k), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    # 5. find contours
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. filter, measure, classify, draw
    out = img.copy()
    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        count += 1
        x, y, w, h = cv2.boundingRect(c)
        name = classify(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(out, f"{name} ({int(area)}px)", (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(out, f"Objects: {count}", (10, out.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return out, clean, count


def main():
    img = cv2.imread(sys.argv[1]) if len(sys.argv) > 1 else make_test_image()
    if img is None:
        sys.exit("Could not read image.")

    win = "Object Counter"
    cv2.namedWindow(win)
    nothing = lambda v: None
    cv2.createTrackbar("Blur", win, 5, 25, nothing)
    cv2.createTrackbar("Threshold (0=auto)", win, 0, 255, nothing)
    cv2.createTrackbar("Min area", win, 500, 5000, nothing)
    cv2.createTrackbar("Morph", win, 3, 15, nothing)

    while True:
        # Stop cleanly if the user closes the window with the X button.
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

        result, mask, count = process(
            img,
            cv2.getTrackbarPos("Blur", win),
            cv2.getTrackbarPos("Threshold (0=auto)", win),
            cv2.getTrackbarPos("Min area", win),
            cv2.getTrackbarPos("Morph", win),
        )
        cv2.imshow(win, result)
        cv2.imshow("Mask (what the computer sees)", mask)
        if cv2.waitKey(30) & 0xFF in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()