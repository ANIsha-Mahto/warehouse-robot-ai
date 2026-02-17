import cv2
import numpy as np

def detect_object(image, show_debug=False):
    """
    Detects objects (boxes) in an image.
    Returns:
    - list of cropped images of objects
    - image with bounding boxes drawn
    - list of bounding boxes (x, y, w, h)
    """
    if isinstance(image, str):
        # If a path is given, read the image
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"Cannot read image at {image}")

    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cropped_list = []
    bboxes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:  # adjust threshold if needed
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = image[y:y + h, x:x + w]
            cropped_list.append(cropped)
            bboxes.append((x, y, w, h))

    if show_debug:
        cv2.imshow("Detected Objects", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cropped_list, output, bboxes


# --- Optional: test with images when run directly ---
if __name__ == "__main__":
    image_path = "sample_images/box1.jpeg"  # Change as needed
    crops, output, boxes = detect_object(image_path, show_debug=True)
    print("Detected boxes:", len(boxes))
