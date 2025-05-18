# 🚗 AI-Driven Car Simulation Using Computer Vision & Deep Learning

![GitHub stars](https://img.shields.io/github/stars/Julian1777/self-driving-project?style=social)

A computer vision system for autonomous driving tasks. This project uses traditional CV methods and deep learning models (CNN, U-Net, YOLO, SCNN) for:

- 🚦 Traffic light detection & classification  
- 🛑 Traffic sign recognition  
- 🛣️ Lane detection (Hough Transform & SCNN)  
- 🧠 Multi-model inference and simulation using Pygame  

## 🎥 Demo

| Lane Detection | Sign Recognition | Traffic Light Detection |
|----------------|------------------|--------------------------|
| ![lane](assets/lane.gif) | ![sign](assets/sign.gif) | ![light](assets/light.gif) |


## 🔧 Features

-  Lane detection with SCNN and OpenCV (comparison)
-  Traffic sign classification using CNN
-  Traffic light detection (YOLO) + classification
-  Video-based inference pipeline
-  Multi-window simulation using Pygame
- 🚀 Coming soon: CARLA integration & real-time testing

## 🛠️ Built With

- TensorFlow / Keras
- OpenCV
- YOLOv8 (Ultralytics)
- Python
- Pygame
- CARLA (planned)

## 📚 Datasets Used

- **CU Lane Dataset** for lane segmentation
- **LISA Traffic Sign Dataset** for sign classification
- **DLDT / LISA** for traffic light classification & detection
- **Mapillary** for sign detection
- **BDD** for vehicle and pedestrian detection

## 📊 Results

| Model        | Task                               | Accuracy / IoU | Dataset   |    Size    | Epochs   |
|--------------|------------------------------------|----------------|-----------|------------|----------|
| CNN          | Sign Classification                | 89%            | GTRSB     |            |20        |
| Yolov8       | Sign Detection                     | 89%            | Mapillary |            |50        |
| Yolov8       | Traffic light Light Detection      | mAP x          |           |            |50        |
| SCNN         | Lane Detection                     | IoU x          | Culane    |            |x         |
| SCNN         | Vehicle & Pedestrian detection     | IoU x          | BDD       | 100k       |30        |


## 🛣️ Roadmap

- [x] Sign classification (CNN)
- [x] Traffic light classification
- [x] Lane detection (U-Net, SCNN, Hough)
- [ ] Integrate all models into Pygame
- [ ] Complete CARLA test scenario
