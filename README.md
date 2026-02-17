**Intelligent Object Recognition and Query System for Warehouse Robotics**

**Overview**
1. This project is a prototype system for a warehouse robot that can:
2. Identify and locate objects from warehouse images
3. Classify objects into categories such as fragile, heavy, and hazardous
4. Retrieve relevant handling instructions using a Retrieval-Augmented Generation (RAG) system

The system integrates Computer Vision, Machine Learning, and Knowledge Retrieval into a modular AI pipeline.

**Part 1: Computer Vision Module**
**Approach**
    Edge detection and contour analysis using OpenCV
    Convert images to grayscale and apply Gaussian blur
    Detect object boundaries using Canny edge detection
    Filter contours based on area (threshold = 2000 pixels) to remove noise

For each valid object, the system calculates:

    Bounding box
    Dimensions in pixels
    Center coordinates

The system overlays bounding boxes, dimensions, and center coordinates on the output image.

**Limitations**

    Difficulty with overlapping objects
    Sensitive to poor lighting conditions
    Works best with structured objects (boxes, pallets)

**Part 2: Machine Learning Module**
**Model**
ResNet18 fine-tuned on a small warehouse dataset (~100 images)

**Classes**
Fragile
Heavy
Hazardous

**Performance**

Approximately 82% classification accuracy.

**Limitations**

Confusion between heavy and hazardous due to visual similarities
Assumes cropped and centered objects
Requires a larger dataset for production-level deployment

**Part 3: RAG System**
The system uses a small knowledge base containing 10–15 documents, including:

Handling instructions
Safety protocols
Equipment specifications
Troubleshooting guides

It uses embedding and retrieval techniques to answer natural language queries.

**Example Queries**

"How should the robot handle fragile items?"
"What safety checks are needed before moving hazardous materials?"

**Part 4: Integration**
This module combines the Computer Vision, Machine Learning, and RAG components.

For a given input image:

Detects objects
Classifies them
Retrieves relevant handling instructions

**Output**

Image with bounding boxes
Predicted object labels
Retrieved safety instructions

**Setup Instructions**

1. Clone this repository
    git clone <your-repo-link>
    cd warehouse-robot_ai

2. Create and activate virtual environment
    python -m venv venv
    .\venv\Scripts\activate   # Windows


3. Install dependencies
    pip install -r requirements.txt

4. Ensure required files are placed correctly

    part1_cv/sample_images/ → Sample images
    part2_ml/model.pth → Trained ML model
    part3_rag/docs/ → Knowledge base documents
    results/ → Output images

**How to Run Each Component**
Part 1 – Object Detection
    python part1_cv/object_detection.py

This will:

Detect objects
Draw bounding boxes
Show dimensions and center coordinates

**Part 2 – Machine Learning Classification**
      python part2_ml/classifier.py


This will:

Load trained ResNet18 model
Predict object class (fragile / heavy / hazardous)
Print prediction result

**Part 3 – RAG System (Query Mode)**
     python part3_rag/query_demo.py


You can enter queries such as:
How should fragile items be handled?

The system will retrieve relevant documents and generate a contextual response.

**Full System Integration **
      python test_integration.py


This will:

Detect objects
Classify them
Retrieve handling instructions
Display the final integrated output

**Dependencies**

Python 3.13
OpenCV (opencv-python)
Torch
TorchVision
Sentence-Transformers
scikit-learn
NumPy
PIL
(Exact versions listed in requirements.txt)

**Challenges Faced and Solutions**

1. Object detection in cluttered frames
Used contour filtering and area thresholding to reduce noise.

2. Model confusion between classes
Improved dataset quality and applied data augmentation techniques.

3. Integrating RAG system with classification output
Ensured query_rag() returns clean string output for smooth integration.

**Results**

Accurate bounding box detection with center coordinates and dimensions
Reliable classification of warehouse objects
Context-aware safety instruction retrieval
Successful integration of CV + ML + RAG into a unified pipeline
