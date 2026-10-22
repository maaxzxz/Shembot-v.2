# Public Counter "Social Distance / Crowd-Count" Barometer

An automated, non-intrusive computer vision system designed to monitor room capacity in waiting areas, clinics, and service desks. Using a repurposed smartphone IP camera, the system processes a live video feed to detect and count individuals in real-time. 

When the room exceeds a safe maximum capacity threshold, it automatically triggers a visual alert (red bounding boxes and text overlay) to assist staff in managing crowds without manual headcounts. 

**Key Technologies:**
* **Main Model:** TensorFlow Hub EfficientDet-d0 (pre-trained on COCO 2017)
* **Baseline Fallback:** OpenCV Haar Cascade (Frontal Face)
* **Dataset:** Kaggle "Crowd Counting" Shopping Center Dataset
* **Input:** Wireless local network stream (IP Webcam/DroidCam)
