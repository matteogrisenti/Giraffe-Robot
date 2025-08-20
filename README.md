# Giraffe Robot - Microphone Positioning System

This project implements a ceiling-mounted giraffe-like robot designed to position a microphone in front of a person in a 5 × 12 meter conference room.

## 📐 Project Description

- Ceiling-mounted robot (4 m high)
- Covers a 5 m × 12 m area
- 5 Degrees of Freedom:
  - 1 spherical joint at base (2 intersecting revolute joints)
  - 1 prismatic joint (to extend downward)
  - 2 revolute joints to orient the microphone
- Microphone orientation fixed to a pitch of 30° for comfortable speaking

## 📁 Repository Structure
```graphql
giraffe_robot/
├── urdf/               # URDF and mesh definitions
├── launch/             # ROS launch files
├── scripts/            # Python control scripts
│   ├── computed_torque_control/       # Modules for torque control
│   ├── dynamics/                      # Modules for dynamic simulation
│   ├── kinematics/                    # Modules for custom kinematics
│   ├── polynomial_trajectory/         # Modules for polynomial trajectory
|   |
│   ├── __init__.py                    # Marks this directory as a Python package
│   ├── chair_marker_publisher.py      # Publishes chair markers in RViz
│   ├── config.py                      # Robot configuration parameters
│   ├── joint_state_publisher.py       # Publishes robot joint states
│   ├── polynomial_trajectory_marker_publisher.py # Publishes trajectory markers
├── config/             # Joint limits, controllers
├── docs/               # Design sketches, notes
├── README.md
├── .gitignore
```

## 📦 Requirements

Make sure you are running **Ubuntu 20.04** with **ROS Noetic**.

## ▶️​ Installing
### 1. Install system dependencies

```bash
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-catkin-tools \
    ros-noetic-desktop-full \
    ros-noetic-joint-state-publisher-gui \
    ros-noetic-robot-state-publisher \
    ros-noetic-xacro \
    ros-noetic-rviz \
    git
```

### 2. Install Pinocchio
```bash
sudo apt install lsb-release curl
sudo mkdir -p /etc/apt/keyrings
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc | sudo tee /etc/apt/keyrings/robotpkg.asc

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" | sudo tee /etc/apt/sources.list.d/robotpkg.list

sudo apt update
sudo apt install robotpkg-py38-pinocchio
```
Add these lines to your ~/.bashrc:
```bash
export PATH=/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/openrobots/lib/python3.8/site-packages:$PYTHONPATH
export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH
```
If are working with WSL in Windows add also: 
```bash
export LIBGL_ALWAYS_INDIRECT=0
export LIBGL_ALWAYS_SOFTWARE=1
```

### 3. Clone and build the workspace
```bash
cd ~
mkdir -p giraffe_ws/src
cd giraffe_ws/src
git clone https://github.com/matteogrisenti/giraffe_robot.git
cd ..
catkin build
```
Then source the workspace:
```bash
source devel/setup.bash
```

### 4. Preparation scripts
Make the package's Python nodes executable:
```bash
chmod +x ~/giraffe_ws/src/giraffe_robot/scripts/*.py

## 🚀 Launch
To visualize the robot:
```bash
roslaunch giraffe_robot visualize.launch
```

## 📚 Assignment Requirements

In addition to visualizing the robot in RViz, some tests are required to verify the kinematics, dynamics, and control capabilities.

### 1) Custom Kinematics Test
To test custom kinematics:
```bash
cd ~/giraffe_ws/src/giraffe_robot
python3 scripts/kinematics/test_custom_kinematics.py
```

### 2) Dynamic Simulation
To test dynamic simulation:
```bash
cd ~/giraffe_ws/src/giraffe_robot
python3 scripts/dynamics/test.py
```

### 3) Polynomial Trajectory
To test polynomial trajectory:
```bash
cd ~/giraffe_ws/src/giraffe_robot
python3 -m scripts.polynomial_trajectory.test
```

### 4) Computed Torque Control
To test computed torque control:
```bash
cd ~/giraffe_ws/src
python3 -m giraffe_robot.scripts.computed_torque_control.test
```



## 📜 Mension
- Refernece Git Repository: [text](https://github.com/idra-lab/intro_robotics_labs/blob/main/scripts/L1_1_kinematics.py)