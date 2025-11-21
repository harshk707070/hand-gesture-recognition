# 🖐️ Hand Gesture Recognition AI

Real-time hand-gesture detection using a deep-learning backend + MediaPipe frontend.  
Built end-to-end — dataset creation → augmentation → training → real-time inference.

---

## 🚀 Project Overview

This project is a full-stack AI system that recognizes hand gestures in real-time using a camera.

It contains:

- A custom trained EfficientNet-B0/B2 model for classification  
- A FastAPI backend to load and run predictions  
- A React + MediaPipe frontend with a clean “AI-startup style” UI  
- Fully documented dataset workflows (raw → augmented → final)  
- Clear separation of backend, frontend, and model training pipelines  

It mimics an actual industry workflow: **collect → preprocess → augment → train.**

---

## 🔥 Features

### 🎦 Frontend
- Beautiful “premium AI startup” UI  
- Real-time webcam feed  
- FPS counter  
- Model load status  
- Predict button  
- Retake image option  
- Clean, responsive card layout using Tailwind  

### 🧠 Backend
- FastAPI server  
- EfficientNet-B0/B2 inference  
- Automatic GPU/CPU fallback  
- JSON class map loading  
- Supports CORS for frontend communication  

### 🗂️ Dataset Engineering
- Custom dataset collected using a self-built data collection tool  
- Three datasets:
  - `dataset/` – raw collected images  
  - `dataset_aug/` – programmatically augmented  
  - `dataset_full/` – final cleaned + labeled dataset used for training  

### 🎓 Training Pipeline
- Uses PyTorch + EfficientNet  
- Automatically loads augmented dataset  
- Exports model weights:
  - `efficient_b0.pth`  
  - `efficient_b2.pth`  
- Class map saved as `class_map.json`  
- Supports retraining for higher accuracy  

---

## 🏗️ Repository Structure
```bash
hand-gesture-recognition/
│
├── backend/
│   ├── app.py              # FastAPI application (main server)
│   ├── model_loader.py     # Loading EfficientNet model + class map
│   ├── requirements.txt     # Backend dependencies
│   └── model/
│       ├── efficient_b0.pth
│       ├── efficient_b2.pth
│       └── class_map.json
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx         # React UI
│   │   ├── HandLandmarker.ts
│   │   ├── App.css
│   │   └── index.css
│   ├── vite.config.ts
│   └── package.json
│
├── data_collection/
│   └── capture_images.py
│
├── dataset/                # Raw dataset
├── dataset_aug/            # Augmented dataset
├── dataset_full/           # Final curated dataset
│
├── training/
│   ├── train.py
│   ├── augment_create.py
│   ├── dataset.py
│   ├── model.py
│   └── helpers.py
│
├── README.md
└── .gitignore
```

## 📸 Dataset Creation Workflow
### 1️⃣ Data Collection
```bash
python data_collection/capture_images.py

Gestures collected:
✊ fist
✌️ peace
✋ stop
👍 thumb up
👎 thumb down

Saved in: dataset/
```

### 2️⃣ Data Augmentation
```bash
python training/augment_create.py

Creates:
- Rotations
- Noise
- Zoom
- Brightness changes

Stored in dataset_aug/.

### 3️⃣ Final Dataset
A merged and cleaned version of raw + augmented → stored in dataset_full
```
### 🤖 Model Training
```bash
cd training
python train.py

Outputs:

- efficient_b0.pth
- efficient_b2.pth
- class_map.json

Backend automatically uses these files.

```

### 🧪 Backend Setup & Run (FastAPI)
```bash
- Install dependencies:
cd backend
pip install -r requirements.txt

- Start server:
uvicorn app:app --reload

- Backend runs at:
http://127.0.0.1:8000
```

### 🎨 Frontend Setup & Run (React + Vite)
```bash
- Install dependencies:
cd frontend
npm install

- Start UI:
npm run dev

- Frontend runs at:
http://localhost:5173
```
### 🛠️ Full Project Commands Summary
```bash
- Activate venv
venv\Scripts\activate

- Frontend
cd frontend
npm install
npm run dev

- Backend
cd backend
pip install -r requirements.txt
uvicorn app:app --reload

- Training
cd training
python train.py

- Data collection
python data_collection/capture_images.py

```
