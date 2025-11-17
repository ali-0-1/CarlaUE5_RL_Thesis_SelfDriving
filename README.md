
# Reinforcement Learning with PPO in CARLA UE5

## Overview
This project is based on a bachelor's thesis related to Reinforcement Learning using the PPO algorithm in the CARLA UE5 Town10 simulation environment.

### Thesis Details
- **Topic:** Reinforcement Learning Algorithm Development via CARLA UE5 Simulator  
- **Implementation:** Steering via Local Planner, Brake and Throttle via Agent  
- **Status:** Thesis will be published once procedures are completed

---
## Workstation Specs

| Component | Details                |
|-----------|------------------------|
| GPU       | NVIDIA RTX A4000       |
| Memory    | 64 GB                  |
| OS        | Ubuntu 22.04           |
| CARLA     | UE5-dev → 0.10.0       |

---

## How to set the project folder?

* Download packaged version: https://github.com/carla-simulator/carla/releases/tag/0.10.0
* Create a virtual environment
* * python3 -m venv venv
* * source venv/bin/activate

Installations:
* install the CARLA wheel according to your python version via *pip* from *\PythonAPI\carla\dist* path
* pip install carla-<version>.whl
* Adjust CUDA version as needed
* pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
* pip install stable-baselines3[extra]
* pip install gymnasium==0.29.1
* pip install keras==2.10.0
* pip install matplotlib==3.9.4
* pip install tf2onnx onnx onnxruntime onnx2pytorch #to convert keras model into troch
* pip install tensorflow==2.15.0
* pip install numpy==1.24.3
* pip install nvidia-pyindex nvidia-tensorrt
* pip install --upgrade matplotlib # if necessary upgrade matplotlib, due to some warnings 

