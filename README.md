# iRobot Simulator

This is a simulator for the iRobot Create 2 model.

## Installation

For this installation, have the AutonomyLabs (https://github.com/AutonomyLab/create_autonomy) package installed and on your ROS path.

Create a catkin workspace, and clone the repo:
```
mkdir irobot_simulator/src -p
cd irobot_simulator/src
git clone git@github.com:JamesUnicomb/irobot_simulator.git
cd ..
catkin build
```



## Reinforcement Learning

To launch a basic maze world:
```
roslaunch irobot_simulator create_2_simulator.launch
