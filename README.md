# 📘 Vision Agent – System Design

This document explains the architecture, model choices, trade-offs, drawbacks, LPR fixes, and scaling strategy for the Vision Agent system.

---

## 1. Architecture — How Data Flows from Video to Answer

The system contains two major components:

### A. Perception Pipeline (ingest.py)

1. Video Input  
   Uses OpenCV to read the video and process only every 5th frame to reduce compute cost.

2. Object Detection (YOLOv8m)  
   Detects:  
   - Cars  
   - Trucks  
   - Buses  
   - Motorcycles  
   - Persons  
   YOLO assigns track IDs to follow the same object across frames.

3. License Plate Detection + OCR
   - A YOLO-based license plate detector extracts plate regions
   - I am using the model from Hugging Face  
   - Only the plate crop is sent to EasyOCR  
   - OCR returns alphanumeric text

5. Metadata Extraction  
   For each detected object:  
   - Timestamp  
   - Object type  
   - Bounding box  
   - Confidence  
   - Color  
   - License plate text  
   Saved as output/metadata.json.

---

### B. Query Agent (agent.py)
CityEye: Video Metadata Intelligence Agent

**CityEye** is an intelligent SQL-powered video surveillance assistant designed to answer natural-language queries about vehicle and object metadata extracted from video streams.  
This component forms the *analytics and query layer* of the overall VMS (Video Management System) pipeline.

---
## 📌 Overview
The goal of this module is to:

- Convert raw per-frame detection JSON into a structured **SQLite database**  
- Expose that database to an **LLM-based agent** using LangChain  
- Allow the user to ask **natural-language questions** such as:
  - “When did the red car appear?”
  - “Show all trucks with visible license plates.”
  - “Find the timestamp of a blue motorcycle.”
- I had used ollama3 which is hosted locally in laptop.

CityEye interprets queries, generates optimized SQL, executes it on the metadata DB, and provides clean professional answers.

---

## 🧱 Architecture


---

### 📌 Summary of Data Flow

Video → YOLO → LP Detector → OCR → JSON → SQLite → Qury → Agent  → LLM → Answer

---

## 2. Model Selection — Why These Models?

### YOLOv8m
- Accurate + real-time  
- Supports tracking  
- Works well for surveillance videos  
- Completely open-source

### YOLO LP Detector
- Best for detecting small, rectangular objects (plates)  
- Fast + robust + open-source

### EasyOCR
- Reliable on noisy real-world text  
- GPU support  
- Open-source  
- No external API required  

### SQLite + Llama3
- Very lightweight metadata storage  
- Fast SQL queries  
- Local LLM → free + private

---

## 3. Trade-offs — Speed vs. Accuracy

### Frame Sampling
- Processing every 5th frame increases speed  
- Minor accuracy loss for short-lived objects

### Model Size
- YOLOv8m = balanced accuracy & performance  
- Smaller LP detector = faster inference
- Yolov1X.pt huggingface public model for recognizing number plates.

### OCR
- EasyOCR is fast but slightly less accurate  
- Preprocessing helps improve reliability

### LLM Query Parsing
- Llama3 is fast  
- Use SQLDatabase from langchain to make operations on database by LLM itself without human interaction and check in loop process for errors.

## 4. Drawbacks / Limitations

### ❗ License Plate Recognition Is Not 100% Accurate
During testing, the system sometimes extracted incorrect license numbers due to:
- Motion blur  
- Low resolution  
- Skewed/angled plates  
- Non-standard fonts  
- Misreading characters (0/O, 1/I, 5/S etc.)
You can see in this image ![LPR in one frame](https://raw.githubusercontent.com/<username>/<repo>/main/images/diagram.png)
### ❗ Reason:
- EasyOCR was not specifically pretrained on the exact style of plates in the video.  
- LP detector sometimes captures extra non-plate text.  
- Time constraints prevented model fine-tuning.

---

## 5. How to Fix the LPR Limitation (Path to Rectify)


### LPR Strategy (Crucial Optimization)

#### ✅ Current Approach (Prototype Stage)
You run OCR on the entire vehicle crop.  
This works, but may accidentally read text that is not a license plate  
(e.g., “FedEx”, “Civic”, stickers, bumper text).

#### ❗ The Issue
- Running OCR on a full vehicle is computationally expensive  
- OCR may extract unrelated text  
- License plates are small → model needs focused region  
- Leads to mismatches like wrong or partial numbers

#### ⭐ Recommended Production Pipeline
In a real “CityEye” system, the correct LPR pipeline is:

1. Vehicle Detection (YOLO)  
2. License Plate Detection (specialized YOLO model trained for rectangular plates)  
3. OCR (PaddleOCR / EasyOCR)  
   👉 Run ONLY on the tiny plate crop, not the full vehicle

This significantly improves:
- Accuracy  
- Speed  
- Reliability  
- Reduces false-text detection  

#### 🕒 Why It’s Not Used Here
Due to the assignment's tight time limit , full LPR retraining or fine-tuning wasn't possible.

#### 🛠 Future Improvement
To achieve near-perfect LPR accuracy:
- Retrain YOLO plate detector on real dataset  
- Fine-tune OCR model on local plate fonts  
- Add multi-frame OCR voting (consensus from multiple frames)  
- Apply regex validation to reject invalid plate formats  

This is the exact “path to rectify” the drawback.

---

## 6. Scaling — Handling 100+ Live Camera Feeds

### A. Distributed Microservice Architecture
1. Ingestion Workers  
   - 2–4 camera streams per worker  
   - Autoscaling supported

2. GPU Detection Cluster  
   - YOLO + LP detector + OCR run inside inference servers  
   - Frames are queued through Kafka/RabbitMQ

3. Central Metadata Database  
   Replace SQLite with:  
   - PostgreSQL  
   - Qdrant/ChromaDB for visual search

4. Query/LLM Service  
   - Converts user queries → SQL  
   - Returns structured answers

---

### B. GPU Scaling
- Each GPU handles ~20–25 video streams at 2 FPS  
- 4 GPU nodes → 80–100 streams  
- Kubernetes HPA manages scaling automatically

---

### C. Fault Tolerance
- If a worker fails, streams shift to another worker  
- Centralized storage prevents metadata loss  
- System continues uninterrupted

---

## ✔ Final Notes

The system balances:  
- Real-time performance  
- Open-source tools  
- Practical accuracy  
- Scalable design  

The main limitation is LPR accuracy, which can be greatly improved using the recommended production pipeline above.
