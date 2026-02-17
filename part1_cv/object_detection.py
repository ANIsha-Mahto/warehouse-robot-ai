import cv2
import numpy as np
import os

# Paths
input_folder = "part1_cv/sample_images"
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

def detect_object(image_path, show_debug=False):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found at", image_path)
        return None, None, None

    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Adaptive threshold to find bright regions ---
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 5
    )

    # Dilate to merge regions
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, output, None

    # Pick the largest contour (assume this is the object)
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    # Draw bounding box
    cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
    center_x, center_y = x + w//2, y + h//2
    cv2.circle(output, (center_x, center_y), 5, (0,0,255), -1)
    cv2.putText(output, f"W:{w}px H:{h}px", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cropped = image[y:y+h, x:x+w]

    if show_debug:
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Dilated", dilated)
        cv2.imshow("Detected Object", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cropped, output, (x, y, w, h)

# Optional batch processing
if __name__ == "__main__":
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            cropped, result, bbox = detect_object(image_path)
            output_path = os.path.join(output_folder, f"output_{filename}")
            cv2.imwrite(output_path, result)
            print(f"Processed and saved: {output_path}")
    print("All images processed successfully.")
