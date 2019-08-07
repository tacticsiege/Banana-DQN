# Banana-DQN

This project trains an AI controlled agent to navigate a 3D environment to collect yellow bananas, and avoid the much-hated-by-AI blue bananas. The agent uses a Deep Q-learning Network (DQN) and learns by interacting with the environment, without prior knowledge or any hand crafted banana specific code.

![Trained agent in environment](solved_naners.gif)

## Environment Overview

The environment contains a single agent, navigating in a room with randomly placed bananas. The agent can choose which direction to move, and receives feedback from the environment based on the observed state and reward signal. The agent is rewarded for hitting yellow bananas, and punished for colliding with blue bananas.

### State Space

The state space has 37 dimensions representing the agent's observation of the environment.  The values are continuous, and represent the agent's velocity and ray-based perception of objects centered around the agent's forward vector.

### Action Space

The agent is able to choose one of four discrete actions to send to the environment. The discrete actions represent move forward, move backward, turn left, and turn right.

### Rewards and Scoring

The rewards for this environment focus on collecting yellow bananas and avoiding blue bananas.

* ```+ 1.0``` for every yellow banana the agent collects

* ```- 1.0``` for every blue banana the agent collects

Scoring for this single agent episodic task is done by taking the average episode score over 100 consecutive episodes.

For this project to be considered solved, the agent must achieve an average score over 100 consecutive episodes of +13.

## Getting Started

Setup the project with the ```git clone``` -> ```conda create [virtualenv]``` -> ```pip install [dependencies]``` workflow, outlined below in detail.

Additionally you'll need to download the Unity Environment solved in this project, links provided below.

### Installation

1. clone the repository:

    ``` bash
    git clone https://github.com/TacticSiege/Banana-DQN
    cd Banana-DQN
    ```

2. Create a virtualenv with Anaconda and activate it:

    * Linux or Mac:

    ``` bash
    conda create --name drlnd python=3.6
    source activate drlnd
    ```

    * Windows:

    ``` bash
    conda create --name drlnd python=3.6
    activate drlnd
    ```

3. Install project dependencies:

    ``` bash
    cd python/
    pip install .
    cd ..
    ```

4. Download the Unity Environment for your OS:

    | Operating System | Link |
    |------------------|------|
    | Windows (32bit) | [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip) |
    | Windows (64bit) | [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip) |
    | MacOS | [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip) |
    | Linux | [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip) |

    Extract the archive into the project directory, or you may update the ```env_path``` in the notebook(s) to use a custom directory.

5. (Optional) Associate a Jupyter Notebook kernel with our virtualenv:

    ``` bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

    If you forget to activate your virtualenv, you can choose the kernel created above from the Jupyter Notebook UI.

6. Run Jupyter Notebook and explore the repo:

    ``` bash
    jupyter notebook
    ```

### Running the Agent

* Train the agent using ```TrainAgent.ipynb```

* Watch a trained agent by loading saved model weights in the last few cells of the notebook.  This repo contains saved model weights for the solution agent, ```Agent Naners```, that are already setup to run.

See ```Report.md``` for more details on implementation and training results.
