# Adaptive Machine Learning Object-Positioning System (Monocular Vision)

## 📌 Project Overview
This project presents a real-time AI-based object positioning system using a single camera (monocular vision). It detects household objects using YOLOv8 and estimates their distance from the camera using geometric principles and known object dimensions.

## 🎯 Key Features
- Real-time object detection using YOLOv8
- Monocular distance estimation
- Object tracking with persistent IDs
- CSV logging of detected objects
- Lightweight and cost-effective solution

## 🧠 Technologies Used
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- ByteTrack
- Matplotlib

## ▶️ How to Run the Project
### 1️⃣ Clone the Repository
git clone https://github.com/your-username/adaptive-monocular-object-positioning.git
cd adaptive-monocular-object-positioning

### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Run the System
python src/main.py
- Press Q to stop the program
- Output data will be saved as a CSV file


## 📊 Output
● Live webcam feed with:
  - Bounding boxes
  - Object labels
  - Unique IDs
  - Estimated distances (in meters)

● CSV file containing:
  - Frame number
  - Timestamp
  - Object ID
  - Object label
  - Estimated distance

## 📌 Applications
- Smart home automation
- Robotics navigation and obstacle awareness
- Low-cost surveillance systems
- Assistive technologies for visually impaired users
- AI research and academic projects

## 🔮 Future Enhancements
- Improve accuracy using camera calibration techniques
- Custom dataset training for domain-specific objects
- Mobile and edge-device deployment
- Voice-based feedback system
- Integration with IoT and robotics platforms
