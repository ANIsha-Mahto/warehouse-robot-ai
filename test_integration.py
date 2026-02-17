from part1_cv.object_detection import detect_object
from part2_ml.evaluate import predict_image
import cv2
import os

# --- Step 0: Setup ---
input_folder = r"C:\Users\ASUS\Desktop\warehouse-robot_ai\part1_cv\sample_images"
output_folder = r"C:\Users\ASUS\Desktop\warehouse-robot_ai\results"
os.makedirs(output_folder, exist_ok=True)

# --- Step 3: RAG Knowledge Base (demo version) ---
rag_knowledge_base = {
    "fragile": "Handle with care. Avoid dropping or shaking. Use gloves if needed.",
    "heavy": "Use mechanical assistance. Do not lift manually. Ensure robot gripper supports weight.",
    "hazardous": "Use protective gloves and safety equipment. Avoid sparks or heat sources. Follow safety protocol X.",
}

# --- Process all images in the folder ---
for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    image_path = os.path.join(input_folder, filename)
    print(f"\nProcessing: {filename}")

    # --- Step 1: Detect object ---
    cropped, output, bbox = detect_object(image_path, show_debug=False)

    if cropped is None:
        print("No object detected.")
        continue

    # --- Step 2: Classify object ---
    label = predict_image(cropped)
    print("Predicted Class:", label)

    # --- Step 3: Retrieve handling instructions ---
    handling_instructions = rag_knowledge_base.get(label.lower(), "No instructions available for this object.")
    print(f"Handling Instructions: {handling_instructions}")

    # --- Step 4: Overlay class and instructions on image ---
    text_y = 30
    cv2.putText(output, f"Class: {label}", (10, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    text_y += 40
    for line in handling_instructions.split('. '):
        cv2.putText(output, line.strip(), (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        text_y += 30

    # --- Step 5: Save output image ---
    output_path = os.path.join(output_folder, f"output_{filename}")
    cv2.imwrite(output_path, output)
    print(f"Saved detected object image to: {output_path}")

print("\nAll images processed successfully!")
