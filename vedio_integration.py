import cv2
from part1_cv.object_detection import detect_object
from part2_ml.evaluate import predict_image
from part3_rag.rag_system import query_rag
import os

# --- Video input/output ---
input_video = r"C:\Users\ASUS\Desktop\warehouse-robot_ai\part1_cv\sample_images\warehouse_video.mp4"
output_video = r"C:\Users\ASUS\Desktop\warehouse-robot_ai\results\tracked_output.mp4"

cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error opening video file. Check path:", input_video)
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# --- Main loop ---
frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    # Detect objects every 10 frames
    if frame_num % 10 == 1:
        try:
            cropped_list, detected_frame, bboxes = detect_object(frame)
        except ValueError:
            cropped_list, detected_frame, bboxes = [], frame.copy(), []

        labels = []
        instructions_list = []

        for cropped in cropped_list:
            label = predict_image(cropped)
            instructions = query_rag(f"How should the robot handle {label} items?")
            labels.append(label)
            instructions_list.append(instructions)

    # Overlay boxes, labels, and instructions
    output = frame.copy()
    line_spacing = 20  # space between lines
    object_spacing = 50  # vertical space between different objects' text

    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if i < len(labels):
            label = labels[i]
            instructions = instructions_list[i]

            # Overlay label above the box
            cv2.putText(output, f"Class: {label}", (x, y - object_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Overlay instructions, each object spaced vertically
            text_y = y - object_spacing + line_spacing
            if isinstance(instructions, str):
                lines = instructions.split('. ')
            elif isinstance(instructions, tuple) or isinstance(instructions, list):
                lines = list(instructions)
            else:
                lines = []

            for line in lines:
                cv2.putText(output, line.strip(), (x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                text_y += line_spacing


    out.write(output)
    cv2.imshow("Tracking Demo", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video processing complete! Saved to {output_video}")
