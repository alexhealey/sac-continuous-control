# sac-reacher

An implementation of Soft Actor Critic to solve the Unity Machine Learning Agents Toolkit Reacher environment. 

## Introduction

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

![Trained Agent](reacher.gif)

A reward of +0.1 is provided for each timestep where the agent's hand is in the correct location.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

## Installation

To set up your python environment to run the code in this repository, follow the instructions below.

Create (and activate) a new environment with Python 3.6.
Linux or Mac:

    conda create --name drlnd python=3.6
    source activate drlnd

Windows:

    conda create --name drlnd python=3.6 
    activate drlnd

Follow the instructions in this repository https://github.com/openai/gym to perform a minimal install of OpenAI gym.

Clone this repository  (if you haven't already!) then, install several dependencies.

    pip install -r requirements.txt

Create an IPython kernel for the drlnd environment.

    python -m ipykernel install --user --name drlnd --display-name "drlnd"

Before running code in a notebook, change the kernel to match the drlnd environment by using the drop-down Kernel menu.

![Jupyter Kernel](jupyter_kernel.png)


You will also need to download the environment file and unzip it in the project root directory. The environments are (by platform)

* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
* [Windows 32bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
* [Windows 64bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

## Approach

The implementation approach is based on Soft Actor Critic. For more details on this approach see the following [article]( https://towardsdatascience.com/in-depth-review-of-soft-actor-critic-91448aba63d4).

## Running 

In order to use the project open the Jupyter notebook `Report.ipynb`. This notebook contains further details of the environment, agents and training.


