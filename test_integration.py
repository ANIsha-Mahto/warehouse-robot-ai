import os
import sys
import warnings

# --- Step 0: Quiet the technical noise ---
# Suppress specific library warnings for a cleaner layman experience
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Reduce technical output

import cv2
# Add current directory to path just in case
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from part1_cv.object_detection import detect_object
from part2_ml.evaluate import predict_image
from part3_rag.rag_system import query_rag

# --- Setup & Image Selection ---
SAMPLE_DIR = os.path.join("part1_cv", "sample_images")

def select_image():
    # Find all valid images in the directory
    valid_extensions = (".jpg", ".jpeg", ".png")
    images = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(valid_extensions)]
    
    if not images:
        print(f"Error: No images found in {SAMPLE_DIR}")
        exit(1)

    # CLI Support: Check if user passed an image name as an argument
    if len(sys.argv) > 1:
        img_name = sys.argv[1]
        if img_name in images:
            return os.path.join(SAMPLE_DIR, img_name)
        else:
            print(f"Error: '{img_name}' not found. Available: {', '.join(images)}")
            exit(1)

    # Interactive Menu
    print("\n" + "="*50)
    print("      WAREHOUSE ROBOT AI - IMAGE SELECTION")
    print("="*50)
    for i, img in enumerate(images, 1):
        print(f"{i}. {img}")
    print("Q. Quit")
    print("="*50)

    while True:
        choice = input("\nSelect an image number to analyze (or 'Q'): ").strip().upper()
        if choice == 'Q':
            print("Exiting...")
            exit(0)
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(images):
                return os.path.join(SAMPLE_DIR, images[idx])
        except ValueError:
            pass
        print("Invalid choice. Please pick a number from the list.")

# Get the selected image
image_path = select_image()

print("\n" + "="*50)
print(f" WAREHOUSE ROBOT AI - ANALYSIS START ")
print("="*50)
print(f"Processing: {os.path.basename(image_path)}")

# --- Step 1 & 2: CV and ML ---
# print("Processing your image...")
cropped, output, bbox = detect_object(image_path, show_debug=False)

if cropped is None:
    print("Could not analyze the image. Please check the object placement.")
    exit(1)

label = predict_image(cropped)
print(f"\n[Detected Item Category]: {label.upper()}")

# --- Step 3: RAG Handling Instructions ---
# print("Retrieving safety instructions...")

try:
    # A simple natural question for the RAG system
    question = f"How should a robot handle a {label} item? What are the simple safety rules?"
    handling_instructions = query_rag(question)
except Exception as e:
    handling_instructions = "The safety guide is currently unavailable. Please contact a supervisor."

print("\n--- SAFETY & HANDLING GUIDE ---")
print(handling_instructions)
print("="*50)

# --- Final Save ---
output_folder = "results"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "result_" + os.path.basename(image_path))
cv2.imwrite(output_path, output)

print(f"\nAnalysis complete. Result saved to: {output_path}")
print("="*50 + "\n")
