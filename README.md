# Multi-Object Detection and Tracking using YOLOv8 and DeepSORT

## Project Overview
This project implements a computer vision pipeline for detecting and tracking multiple objects in a video using YOLOv8 for object detection and DeepSORT for tracking. The system assigns a unique ID to each object and maintains identity consistency across frames.

The solution is designed to handle real-world challenges such as motion, occlusion, and overlapping objects.

---

## Objectives
- Detect multiple objects in a video stream
- Assign unique and persistent IDs to each object
- Maintain identity consistency across frames
- Handle occlusion and motion variations

---

## Methodology

### Object Detection
- Model: YOLOv8 (Ultralytics)
- Detects objects in each frame with bounding boxes and confidence scores

### Object Tracking
- Algorithm: DeepSORT
- Components:
  - Kalman Filter for motion prediction
  - Hungarian Algorithm for data association
  - Appearance feature matching for identity preservation

### Pipeline Flow
1. Read input video frame-by-frame  
2. Perform object detection using YOLOv8  
3. Extract bounding boxes and confidence scores  
4. Pass detections to DeepSORT tracker  
5. Assign unique IDs to each object  
6. Draw bounding boxes and IDs on frames  
7. Generate annotated output video  

---

## Output Video

The annotated output video with bounding boxes and tracking IDs can be accessed below:

https://drive.google.com/file/d/1IVT4KVAavtyDxayDyr2ukiF_qxJ_EzV-/view?usp=drivesdk

---

## Results

![Result 1](screenshot1.png)  
![Result 2](screenshot2.png)  
![Result 3](screenshot3.png)

---

## Installation and Setup

### Prerequisites
Ensure the following are installed on your system:
- Python 3.8 or higher  
- pip (Python package manager)  
- Git  

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/Bhoopendrapatel/multi-object-tracking-yolo-deepsort.git
cd multi-object-tracking-yolo-deepsort