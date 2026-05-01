# 📌 ReID-Based Tracking and Occlusion Handler (Simple Physics Experiment)

## 🧪 Project Overview

This project is an **experimental implementation** to explore person detection and tracking with ReID, and how identity behaves during **occlusion using simple physics intuition**. It is not designed as a production system, but as a learning and exploration prototype to understand tracking behavior in real-world-like conditions.

As a developer experiment, the focus is on: detecting people, maintaining identity, and handling temporary disappearance using basic motion assumptions.

## 👁️ Detection and Representation

Each frame is processed using a person detection model to extract bounding boxes.

For each detected person, a **ReID embedding model (CNN-based feature extractor, e.g., OSNet/MobileNet-style embeddings)** is used to represent appearance.

These embeddings help maintain identity consistency across frames.

## 🔗 Identity Matching

Identity matching is performed using **NumPy / SciPy cosine similarity** between embeddings.

The closest match above a threshold is assigned the same ID; otherwise a new ID is created.

## 🧩 Occlusion Handling (Simple Physics Experiment)

When a person becomes **occluded**, the system keeps the identity alive instead of removing it.

For this experiment, we use **simple physics intuition of motion**:

* assume constant velocity
* assume short-term directional continuity

This is used only to **approximate possible continuation during occlusion**, not for precise prediction.

## 📍 Reappearance

When the person reappears, the system compares the new embedding with stored identities. If matched, the same ID is restored, completing:

**visible → occluded (predicted state) → re-identified**

## ⚙️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the program

```bash
python /world_model/main.py
```


### 4. Output

* Live video window with bounding boxes
* Persistent IDs across frames
* Re-identification after occlusion recovery

## 🧠 Summary

This project is a **learning experiment** combining:

* ReID-based appearance matching
* Simple motion physics intuition for occlusion

It demonstrates how identity can be maintained in a lightweight way without complex tracking models, primarily for exploration and understanding of occlusion behavior in computer vision systems.
