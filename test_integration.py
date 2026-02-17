from part1_cv.object_detection import detect_object
from part2_ml.evaluate import predict_image
import cv2

# --- Step 0: Setup ---
# Path to your test image
image_path = r"C:\Users\ASUS\Desktop\warehouse-robot_ai\part1_cv\sample_images\box3.jpeg"

# --- Step 1: Detect Object ---
cropped, output, bbox = detect_object(image_path, show_debug=True)  # show debug windows

if cropped is None:
    print("No object detected.")
    exit()

# --- Step 2: Classify Object ---
label = predict_image(cropped)
print("Predicted Class:", label)

# Display the class on the image
cv2.putText(output, f"Class: {label}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Detected Object", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Step 3: RAG Handling Instructions ---
# For demo purposes, a simple dictionary maps classes to instructions
rag_knowledge_base = {
    "fragile": "Handle with care. Avoid dropping or shaking. Use gloves if needed.",
    "heavy": "Use mechanical assistance. Do not lift manually. Ensure robot gripper supports weight.",
    "hazardous": "Use protective gloves and safety equipment. Avoid sparks or heat sources. Follow safety protocol X.",
}

# Retrieve handling instructions based on predicted class
handling_instructions = rag_knowledge_base.get(label.lower(), "No instructions available for this object.")

print("\n--- Handling Instructions ---")
print(f"Object Class: {label}")
print(f"Instructions: {handling_instructions}")
