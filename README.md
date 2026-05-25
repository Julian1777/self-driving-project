<p align="center">
  <img src="media/bannernobg.png" alt="VisionPilot Banner" height="200" />
</p>

# VisionPilot: Autonomous Driving Simulation, Computer Vision & Real-Time Perception (BeamNG.tech)

<p align="center" style="margin-bottom:0;">
  <img src="media/demo_gifs/combined_demos.gif" alt="Combined demo preview" width="560" />
</p>

## Overview

A modular Python project for autonomous driving research and prototyping, fully integrated with the BeamNG.tech simulator and Foxglove visualization. This system combines traditional computer vision algorithms and deep learning (CNN, YOLO) with real-time sensor fusion and autonomous vehicle control to tackle:

- **Multi-Lane Detection**: YOLOP, Traditional CV
- **Traffic Sign**: Classification & Detection
- **Traffic Lights**: Classification & Detection
- **Object Detection**: Vehicles, pedestrians, cyclists and more
- **Multi-Sensor Fusion**: Camera, Lidar, Radar, GPS, IMU <!-- - **Microservices Architecture**: Containerized multi-model inference (Docker), orchestrated via central aggregator -->
- **Real-Time Control**: Model Predictive Control (MPC) for integrated steering & throttle optimization
- **Visualization**: Real-time monitoring with Foxglove WebSocket + multiple CV windows
- **Configuration System**: YAML-based modular settings
  
## Table of Contents

- [VisionPilot: Autonomous Driving Simulation, Computer Vision \& Real-Time Perception (BeamNG.tech)](#visionpilot-autonomous-driving-simulation-computer-vision--real-time-perception-beamngtech)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [System Architecture \& Data Flow](#system-architecture--data-flow)
  - [Demos](#demos)
    - [Multi-Lane Detection](#multi-lane-detection-stress-testing)
    - [Emergency Braking (AEB) Demo](#emergency-braking-aeb-demo)
    - [Blind Spot Detection (BSD) Demo](#blind-spot-detection-bsd-demo)
    - [Sign Detection \& Detection and classification](#sign-detection--detection-and-classification)
    - [Traffic Light Detection \& Classification Demo](#traffic-light-detection--classification-demo)
    - [Latest Lane Detection Demo (v2)](#latest-lane-detection-demo-v2)
    - [YOLOP Lane Detection Demo](#yolop-lane-detection-demo)
    - [Foxglove Visualization Demo](#foxglove-visualization-demo)
    - [Segmentation Demo](#segmentation-demo)
  - [Sensor Suite](#sensor-suite)
  - [Microservices Architecture](#microservices-architecture)
  - [Roadmap](#roadmap)
  - [Note on Installation](#note-on-installation)
  - [Known Limitations](#known-limitations)
  - [Credits](#credits)
  - [License](#license)



## System Architecture & Data Flow

The following diagram illustrates the complete data flow from the simulation environment through perception, control, and final actuation:

```text
┌─────────────────────────────────────────────────────────────────┐
│                     BeamNG.tech Simulation                      │
│  (Camera, Lidar, Radar, GPS, IMU, Vehicle Speed, Orientation)   │
└───────────────────────────────┬─────────────────────────────────┘
                                │ Sensor Data Stream
┌───────────────────────────────▼─────────────────────────────────┐
│                      Perception Layer                           │
│                                                                 │
│  ┌──────────────┐   ┌───────────────┐   ┌───────────────────┐   │
│  │ CV Lane Det. │   │  YOLOP Model  │   │ Object Detection  │   │
│  │ (Lane Center,│ + │ (Segmentation,│ + │ (Vehicles, Signs, │   │
│  │  Deviation)  │   │  Drivable)    │   │  Traffic Lights)  │   │
│  └──────────────┘   └───────────────┘   └───────────────────┘   │
└───────────────────────────────┬─────────────────────────────────┘
                                │ Waypoints, Lane Metrics, Obstacles
┌───────────────────────────────▼─────────────────────────────────┐
│                  Planning & Control Layer                       │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Model Predictive Control (MPC)               │  │
│  │  - Plans 1-sec trajectory based on exact vehicle state    │  │
│  │  - Optimizes smooth Steering + Throttle simultaneously    │  │
│  │  - Constrained by vehicle dynamics & physical bounds      │  │
│  └───────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────┘
                                │ Proposed Control (Steering, Throttle)
┌───────────────────────────────▼─────────────────────────────────┐
│                     Active Safety Layer                         │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │             Automatic Emergency Braking (AEB)             │  │
│  │  - Monitors continuous Radar TTC (Time-To-Collision)      │  │
│  │  - Overrides MPC throttle limits if collision is imminent │  │
│  └───────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────┘
                                │ Final Actuated Commands
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Vehicle Control (BeamNG.tech)                   │
│                 (Steering, Throttle, Braking)                   │
└─────────────────────────────────────────────────────────────────┘
```

## Demos

### Multi-Lane Detection
Evaluation of the multi-lane perception pipeline across various environmental edge cases, including high-glare transitions, low-light tunnels, and heavy atmospheric fog:

<img src="media/demo_gifs/multi-lane.gif" alt="AEB Demo" width="600" height="337" />

**Extended Demo:** [Watch the full video here](https://youtu.be/IvmJ01pYCSE)

---

### Emergency Braking (AEB) Demo

Watch the Emergency Braking System (AEB) in action with real-time radar filtering and collision avoidance:

<img src="media/demo_gifs/aeb_gif.gif" alt="AEB Demo" width="600" height="337" />

**Extended Demo:** [Watch the full video here](https://www.youtube.com/watch?v=Z8Y2-MpmrRg)

---

### Blind Spot Detection (BSD) Demo
See the Blind Spot Detection (BSD) system in action using radar data to identify vehicles in the blind spot:
<img src="media/demo_gifs/bsd_demo.gif" alt="Blind Spot Detection Demo" width="600" height="337" />
**Extended Demo:** [Watch the full video here](https://www.youtube.com/watch?v=Z8Y2-MpmrRg)

---

### Sign Detection & Detection and classification

This demo shows real-time traffic sign detection and classification:

<img src="media/demo_gifs/sign_demo.gif" alt="Sign Detection Demo & Vehicle Pedestrian" width="600" height="337" />

**Extended Demo:** [Watch the full video here](https://youtu.be/ujGkQJ2BqV0)

> VisionPilot does not yet support multi-camera. This is for demonstration purposes only.

---

### Traffic Light Detection & Classification Demo

This demo shows real-time traffic light detection and classification:

<img src="media/demo_gifs/traffic_light_demo.gif" alt="Traffic Light Detection & Classification Demo" width="600" height="337" />

> No extended Demo avaliable yet.

---

### Latest Lane Detection Demo (v2)

Watch the improved autonomous lane keeping demo (v2) in BeamNG.tech, featuring smoother fused CV+SCNN lane detection, stable PID steering, and robust cruise control:

<img src="media/demo_gifs/lane.gif" alt="Lane Detection Demo" width="600" height="337" />

**Extended Demo:** [Watch the full video here](https://www.youtube.com/watch?v=7eA_XfIkLWQ)

> Note: Very low-light (tunnel) scenarios are not yet supported.

### Previous Lane Detection Demo (v1)

The original demo is still available for reference:

[Lane Keeping & Multi-Model Detection Demo (v1)](https://youtu.be/f9mHigMKME8)

---

### YOLOP Lane Detection Demo
Watch both the raw model segmentation output and the multiple processed lanes on a highway video.

<img src="media/demo_gifs/yolop.gif" alt="YOLOP Lane Detection Demo" width="600" height="337" />

**Extended Demo:** [Watch the full video here](https://youtu.be/CZC2ajqDkuU)

> Note: This is not the final integration of the yolop model in VisionPilot. This only serves as a demo of the model's capabilities and use cases for VisionPilot.

---

### Foxglove Visualization Demo

See real-time LiDAR point cloud streaming and autonomous vehicle telemetry in Foxglove Studio:

<img src="media/demo_gifs/foxglove.gif" alt="Foxglove Visualization Demo" width="600" height="337" />

**Extended Demo:** [Watch the full video here](https://www.youtube.com/watch?v=4HJDvL2Q6AY)

---

### Segmentation Demo

See real-time image segmentation using front and rear cameras:

<img src="media/demo_gifs/segmentation.gif" alt="Segmentation Demo" width="600" height="337" />

**Extended Demo:** [Watch the full video here](https://youtu.be/4PAqcUKqn6c?si=UHw-mw7iLZKGXvav)

---

> More demo videos and visualizations will be added as features are completed.

## Sensor Suite

The vehicle is equipped with a comprehensive multi-sensor suite for autonomous perception and control:

| Sensor                      | Specification                                        | Purpose                                                         |
| --------------------------- | ---------------------------------------------------- | --------------------------------------------------------------- |
| **Front Camera**            | 1920x1080 @ 50Hz, 70° FOV, Depth enabled             | Lane detection, traffic signs, traffic lights, object detection |
| **LiDAR (Top)**             | 80 vertical lines, 360° horizontal, 120m range, 20Hz | Obstacle detection, 3D scene understanding                      |
| **Front Radar**             | 200m range, 128×64 bins, 50Hz                        | Collision avoidance, adaptive cruise control                    |
| **Rear Left & Right Radar** | 30m range, 64×32 bins, 50Hz                          | Blindspot monitoring, rear object detection                     |
| **Dual GPS**                | Front & rear positioning @ 50Hz                      | Localization                                                    |
| **IMU**                     | 100Hz update rate                                    | Vehicle dynamics, pose estimation                               |

<table>
  <tr>
    <td align="center"><img src="media/beamng_images/sensors.png" alt="Sensor Array 1" width="280"/></td>
    <td align="center"><img src="media/beamng_images/radar_front.png" alt="Sensor Array 2" width="280"/></td>
    <td align="center"><img src="media/beamng_images/lidar.png" alt="Sensor Array 3" width="280"/></td>
  </tr>
  <tr>
    <td align="center"><em>Sensor Array</em></td>
    <td align="center"><em>Front Radar</em></td>
    <td align="center"><em>Lidar Visualization</em></td>
  </tr>
</table>

> Configuration files are located in the `/config` directory:

<!-- ## Microservices Architecture

> **Note:** The microservices architecture is documented below as the intended design. **Currently, for active development and rapid iteration, all perception models run locally in-process** (bypassing Docker containers and the aggregator). This allows faster prototyping and validation of the complete pipeline. The containerized microservices will be re-integrated once the core perception, sensor fusion, and control systems are finalized and validated.

VisionPilot is designed to use a **containerized microservices architecture** where each perception task runs as an independent Flask service, orchestrated by a central Aggregator:

### Service Stack (Intended Design)

| Service | Port | Function | Model/Framework |
|---------|------|----------|-----------------|
| **Object Detection** | 5777 | Vehicle, pedestrian, cyclist detection | YOLOv11 |
| **Traffic Light Detection** | 6777 | Traffic light detection & state classification | YOLOv11 |
| **Sign Detection** | 7777 | Traffic sign detection | YOLOv11 |
| **Sign Classification** | 8777 | Traffic sign type classification | CNN |
| **YOLOP** | 9777 | Unified: lanes + drivable area + objects | YOLOPX |

### Data Flow

```
BeamNG Simulation Loop
    ↓
PerceptionClient.process_frame()
    ↓
Aggregator (concurrent orchestration)
    ├─→ Object Detection (5777)
    ├─→ Traffic Light (6777)
    ├─→ Sign Detection (7777)
    ├─→ Sign Classification (8777)
    └─→ YOLOP (9777)
    ↓
Merge all responses
    ↓
Return unified AggregationResult
    ↓
Extract individual results + visualize
```

### Benefits

**Concurrency**: All services run in parallel (ThreadPoolExecutor)  
**Modularity**: Add/remove services without modifying BeamNG code  
**Scalability**: Easy horizontal scaling with container orchestration  
**Fault Tolerance**: Individual service failures don't break the pipeline  
**Reusability**: Services can be used independently or together
-->

## Control Architecture Evolution: PID → MPC

VisionPilot has transitioned from traditional PID-based control to advanced **Model Predictive Control (MPC)** for superior performance:

| Aspect | PID Control (Legacy) | MPC Control (Current) |
|--------|---------------------|----------------------|
| **Strategy** | Reactive (error-based) | Predictive (horizon-based) |
| **Prediction** | None - responds to current error | Looks 1 second ahead |
| **Steering Control** | Separate lateral controller | Integrated optimization |
| **Throttle Control** | Separate cruise control | Integrated optimization |
| **Decision Making** | Independent (steering ≠ throttle) | Simultaneous & coupled |
| **Physics Awareness** | Limited | Full vehicle dynamics model |
| **Comfort** | May be jerky/oscillatory | Smooth by design (cost-weighted) |
| **Obstacle Handling** | Reactive braking only | Proactive path planning |
| **Computational Load** | Low (~1ms) | Medium (~20ms) |
| **Tuning Complexity** | High (multiple PIDs) | Lower (cost matrices Q, R) |

### Why MPC?

MPC fundamentally changes how the vehicle makes decisions:
- **Looks ahead**: Plans the next 1 second of motion
- **Optimizes together**: Steering and throttle decisions are made simultaneously, respecting vehicle physics
- **Respects constraints**: Hard physical limits (steering angle, acceleration) are built in
- **Smooth control**: The cost function naturally penalizes jerky inputs

### Layered Safety Architecture: MPC + AEB

VisionPilot uses a **two-layer safety approach** that combines proactive planning with reactive fallback:

**How it works:**
1. **MPC computes optimal control** considering lane following and smooth acceleration
2. **AEB monitors radar** for imminent collisions:
   - **TTC ≤ 1.0s**: Emergency brake (throttle = 0, acts as safety net)
   - **TTC ≤ 2.5s**: Reduce throttle to 50% (MPC still controls steering for avoidance)

**Benefits:**
- **Proactive**: MPC plans around obstacles smoothly
- **Reactive**: AEB catches any collision MPC didn't anticipate
- **Robust**: Defense-in-depth approach reduces crash risk
- **Future-proof**: When Lidar 3D detection is integrated into MPC obstacles, AEB becomes rarely triggered

## Roadmap

### Perception

- [x] Sign classification & Detection (CNN / YOLO)
- [x] Traffic light classification & Detection (CNN / YOLO)
- [x] Lane detection Fusion (YOLOP / CV)
- [x] 🔥🔥 YOLOP integration
  - [x] Drivable area segmentation
  - [x] Lane detection (segmentation output)
  - [x] Object detection
- [x] CV Lane Detection (Traditional Computer Vision)
- [x] Integrate Majority Voting system for CV
- [x] Lighting Condition Detection
- [x] Real-Time Object Detection (Cars, Trucks, Buses, Pedestrians, Cyclists)
- [ ] 🔥🔥 Speed Estimation using detection from camera and lidar
  - [ ] Multiple Object Tracking (MOT)
- [x] 🔥🔥 Handle dashed lines better in lane detection
- [ ] Road Marking Detection (Arrows, Crosswalks, Stop Lines)
- [ ] 🔥🔥🔥 Lidar Object Detection 3D
- [ ] 💤 Ocluded Object Detection (Detect objects that are partially blocked or not visible in the camera view using radar/lidar)
- [x] Detect multiple lanes
- [ ] 💤 Multi Camera Setup (Will implement after all other camera-based features are finished)
- [ ] 💤 Overtaking, Merging (Will be part of Path Planning)

### Sensor Fusion & Calibration

- [ ] Kalman Filtering
  - [ ] Extended
- [x] Integrate Radar
- [x] Integrate Lidar
- [ ] Integrate GPS
- [ ] Integrate IMU
- [ ] Ultrasonic Sensor Integration
- [ ] 💤💤 SLAM (simultaneous localization and mapping)
  - [ ] Build HD Map of the BeamNG.tech map
  - [ ] Localize Vehicle on HD Map

### Control & Planning

- [x] Integrate vehicle control (Throttle, Steering, Braking Implemented) (PID needs further tuning)
- [x] Integrate PIDF controller
- [x] ⭐ Adaptive Cruise Control (Currently only basic Cruise Control implemented)
- [x] Automatic Emergency Braking AEB (Safety fallback layer for imminent collisions)
  - [ ] Obstacle Avoidance via MPC (Proactive path planning through constraint formulation)
- [x] 🔥 Model Predictive Control MPC (Integrated with CasADi IPOPT solver, replaces PID control)
- [ ] Curve Speed Optimization (Slow down for sharp curves based on lane curvature)
- [ ] Trajectory Prediction for surrounding vehicles
- [x] 🔥 Blindspot Monitoring (Using left/right rear short range radars)
- [ ] Traffic Rule Enforcement (Stop at red lights, stop signs, yield signs)
- [ ] Dynamic Target Speed based on Speed Limit Signs
- [ ] Global Path planning
- [ ] Local Path planning
- [ ] 🔥 Lane Change Logic (MPC)
  - [ ] Check Blindspots before lane change
  - [ ] Signal Lane Change
- [ ] Parking Logic (Path finding / Parallel or Perpendicular)

### Visualization & Logging

- [x] ⭐ Full Foxglove visualization integration (Overhaul needed)
- [x] Modular YAML configuration system
- [x] Real-time drive logging and telemetry
- [ ] 🔥 Birds eye view BEV (Top down view of vehicle and surroundings)
- [ ] Real time Annotations Overlay in Foxglove
- [ ] Show predicted trajectories in Foxglove
- [ ] Show Global and local path plans in Foxglove
- [ ] 💤 Live Map Visualization

> **Note:** Considering moving away from Foxglove entirely to build a custom dashboard. Not a priority at this time.

### Deployment & Infrastructure

- [x] Containerize Models for easy deployment and scalability
  - [x] ⭐ Microservices Architecture (Aggregator + individual services)
  - [x] Message Broker (Redis support in docker-compose)
  - [x] Docker Compose orchestration
  - [x] Aggregator service (concurrent service orchestration)


### README To-Dos

- [x] Add detailed documentation (Lane Det first)
- [x] Add demo images and videos to README
- [ ] 💤💤 Add performance benchmarks section
- [x] Add Table of Contents for easier navigation

### Other

- [x] Vibe-Code a website for the project
- [x] Redo project structure for better modularity

> Driver Monitoring System would've been pretty cool but human drivers are not implemented in BeamNG.tech or Carla

## Legend

> 🔥 = High Priority

> ⭐ = Complete but still being improved/tuned/changed (not final version)

> 💤 = Minimal Priority, can be addressed later

> 💤💤 = Very Low Priority, may not be implement

## Note on Installation

> **Status:** This project is currently in **active development**. A stable, production-ready release with pre-trained models and complete documentation will be available eventually.

## Known Limitations

- **Tunnel/Low-Light Scenarios**: Camera perception fails below certain lighting thresholds
- **Multi-Camera Support**: Single front-facing camera only (future roadmap)
- **PID Controller Tuning**: May oscillate on tight curves
- **Real-World Testing**: Only validated in simulation (BeamNG.tech), for now...

## Credits

**Datasets:**

- CU Lane, LISA, GTSRB, Mapillary, BDD100K

**Simulation & Tools:**

- BeamNG.tech by [BeamNG GmbH](https://www.beamng.tech/)
- Foxglove Studio for visualization
- Docker & Docker Compose for containerization

**Special Thanks:**

- Kaggle for free GPU resources (model training)
- Mr. Pratt (teacher/supervisor) for guidance

## Acknowledgements

**Academic Papers & Research:**

YOLOP/YOLOPX: [Anchor-free multi-task learning network for panoptic driving perception](https://doi.org/10.1016/j.patcog.2023.110152)
```bibtex
@article{YOLOPX2024,
  title={YOLOPX: Anchor-free multi-task learning network for panoptic driving perception},
  author={Zhan, Jiao and Luo, Yarong and Guo, Chi and Wu, Yejun and Liu, Jingnan},
  journal={Pattern Recognition},
  volume={148},
  pages={110152},
  year={2024}
}
```
MPC Controller: [DRL-MPC](https://github.com/ZITingHUANG1/DRL-MPC)
```bibtex
@misc{huang_drlmpc,
  author       = {ZITing Huang},
  title        = {DRL-MPC: Integrating Reinforcement Learning and Model Predictive Control for Enhancing Safety in Automated Vehicle Systems},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  url          = {https://github.com/ZITingHUANG1/DRL-MPC}
}
```

## Citation

If you use VisionPilot in your project, please cite:

```bibtex
@software{visionpilot2026,
  title={VisionPilot: Autonomous Driving Simulation, Computer Vision & Real-Time Perception},
  author={Julian Stamm},
  year={2026},
  url={https://github.com/visionpilot-project/VisionPilot}
}
```

### BeamNG.tech Citation

> **Title:** BeamNG.tech  
> **Author:** BeamNG GmbH  
> **Address:** Bremen, Germany  
> **Year:** 2025  
> **Version:** 0.35.0.0  
> **URL:** https://www.beamng.tech/

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.
