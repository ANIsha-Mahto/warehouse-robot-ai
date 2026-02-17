**Intelligent Object Recognition and Query System for Warehouse Robotics**

**Overview:-**

This project is a prototype system for a warehouse robot that can:
1. Identify and locate objects using its camera
2. Classify objects into categories like fragile, heavy, and hazardous
3. Retrieve relevant handling instructions using a Retrieval-Augmented Generation (RAG) system
4. Optionally, track objects in video and overlay instruction

**Part 1: Computer Vision Module**
Approach-

Edge detection and contour analysis using OpenCV
Convert images to grayscale and apply Gaussian blur
Detect object boundaries using Canny edge detection
Filter contours based on area (threshold = 2000 pixels) to remove noise
For each valid object, calculate:
    Bounding box
    Dimensions in pixels
    Center coordinates
Overlay bounding boxes, dimensions, and center on the output image

Limitations:

    Difficulty with overlapping objects, poor lighting, or complex backgrounds
    Only detects structured objects (boxes, pallets) in this prototype

**Part 2: Machine Learning Module**

Model: ResNet18 fine-tuned on a small warehouse dataset (~100 images)
Classes: fragile, heavy, hazardous
Performance: ~82% accuracy

Limitations:

    Confusion between heavy and hazardous due to visual similarities
    Model assumes cropped and centered objects
    Needs larger dataset and augmentation for production

**Part 3: RAG System**

A small knowledge base with 10–15 documents:

    Handling instructions
    Safety protocols
    Equipment specifications
    Troubleshooting guides
Uses embedding + retrieval to answer natural language queries
Example queries:

    "How should the robot handle fragile items?"
    "What safety checks are needed before moving hazardous materials?"

**Part 4: Integration**

Combines CV, ML, and RAG components

For a given image/video frame:

    1. Detects objects
    2. Classifies them
    3. Retrieves handling instructions

Output: images/videos with bounding boxes, labels, and instructions overlaid
Video tracking:

    Optional feature added using OpenCV’s CSRT tracker
    Tracks objects across multiple frames and overlays instructions dynamically

**Setup Instructions-**

1. Clone this repository:

    git clone <your-repo-link>
    cd warehouse-robot_ai

2. Create and activate virtual environment:

    python -m venv venv
    .\venv\Scripts\activate   # Windows

3. Install dependencies:

    pip install -r requirements.txt

4. Ensure model weights, sample images, and videos are in the correct folders:

    part1_cv/sample_images/ → sample images

    part2_ml/model.pth → trained ML model

    part3_rag/docs/ → knowledge base documents

    results/ → output images/videos

**How to Run Each Component-**

Part 1 – Object Detection (images):

    python test_integration.py


Part 4 – Video Tracking and Full Integration:

    python video_integration.py


RAG System Query (interactive mode):

    python part3_rag/query_demo.py

**Dependencies-**

    Python 3.13
    OpenCV (opencv-python or opencv-contrib-python)
    Torch, TorchVision
    Transformers / Sentence-Transformers (for RAG embeddings)
    scikit-learn, numpy, PIL

(Exact versions in requirements.txt)

**Challenges Faced and Solutions-**

1. Object detection in cluttered frames:

    Used edge detection and contour filtering; optionally adjusted threshold for small objects

2. Tracking objects in video:

    Upgraded to opencv-contrib-python to access CSRT tracker

3. Model confusion between classes:

    Small dataset limitation; added more distinct training examples and data augmentation

4. Text overlap on video frames:

    Split instructions into multiple lines and adjusted vertical spacing

5. Integrating RAG system with dynamic queries:

    Ensured query_rag returns string output (not tuple) to overlay on video

**Results-**

    Detected bounding boxes with center coordinates and dimensions
    Classified objects accurately
    Overlaid RAG instructions on images and videos
    Video tracking demo shows real-time object detection, classification, and instruction overlay

