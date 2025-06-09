#  AgroSight: Intelligent Pest Detection & Advisory

AgroSight is an end-to-end AI-powered system for detecting pests on crop leaves and providing actionable treatment recommendations. The system combines a YOLOv8 object detection model with Gemini LLM integration to deliver pest-specific advice in real time through a user-friendly web interface built in Streamlit.

---

##  Features

-  Detects crop pests from uploaded leaf images
-  Trained on a manually annotated IP102 dataset (5 pest classes)
-  Model built with YOLOv8 (Nano variant)
-  Gemini 2.0 Flash LLM integration for treatment suggestions
-  Interactive Streamlit frontend
-  Metrics: mAP@0.5, mAP@0.5:0.95, Precision, IoU, SSIM, PSNR

---

## Frontend Demo

Below is a snapshot of the AgroSight web interface in action:

![Frontend UI](/frontend.jpg)

Users upload an image → pest is detected → bounding box and name appear → Gemini generates expert advice.

---

##  Model Architecture

- Input: 640x640 RGB image
- Backbone: CSPDarknet (YOLOv8)
- Neck: PANet for multi-scale feature fusion
- Head: Decoupled anchor-free detection
- Output: Bounding boxes + class predictions

Model inference is followed by prompt-based interaction with Gemini Flash to return treatment in JSON format.

---

##  Project Structure

AgroSight/
│
├── backend.py # Inference logic + Gemini prompt
├── frontend.py # Streamlit frontend
├── best.pt # Trained YOLOv8 model weights
├── data.yaml # YOLO config file
├── README.md # This file
└── assets/
└── frontend_output.png 





