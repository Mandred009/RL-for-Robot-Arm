# Learning-to-Push Deep RL for Robot Manipulation

## **Task:** How can we make the robot push the object from one location to the desired location and orientation?

1) Tested Both SAC and PPO algorithms for solving this problem.
2) Environment Created on top of Robosuite.
3) Main task is to push the object to the target green location.
4) SAC algorithm showed better performance compared to PPO.
5) Trained for 6 days on RTX 3090.
6) SAC algorithm pushes the block close enough however still struggles with perfectly aligning with the target place.

## **Detailed Report:** 
https://drive.google.com/file/d/1XyBCPBdta29xblYCtIrkrQSIhehyxUYX/view?usp=sharing

## **Scripts:**
**1) PushAlign.py** - The main environment script containing the reward function, observations and termination checks.

**2) main.py** - Test script to check how the environment is used and how it works.

**3) my_custom_arena.xml** - Custom xml file for Mujoco.

**4) ppo.py** - PPO algorithm implemented from scratch for the environment.

**5) sac.py** - SAC algorithm implemented from scratch for the environment.

**6) test.py** - Script to test the trained RL agent.

## **Demo:** Following is the GIF for the robot attempting to push the block using SAC algorithm.
![Demo Gif](https://github.com/user-attachments/assets/f007ac6d-7bb4-46de-983f-81fa0569e125)
