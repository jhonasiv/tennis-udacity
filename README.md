# Tennis Udacity

In this project, two agents, represented by tennis racket, must learn to hit the ball during a
tennis match. To achieve this goal, the method used
was [Multi-Agent Deep Deterministic Policy Gradient](https://arxiv.org/abs/1706.02275),
implemented with an actor and a critic for each agent.

---

# The environment

The agents receive a +0.1 reward every time it hits the ball, and -0.1 every time the ball hits the
floor or goes out of bounds.

An observation of the environment is composed by a vector with 8 elements, representing the position
and velocity of the ball and the agent. Every time step, the agent is informed with a set formed
from the 3 last observations. This is done to make the agent learn about the concept of movement in
the observations. Each agent performs two continuous actions: moving horizontally and vertically.

This is an episodic task with a continuous action space. An episode ends when the ball hits the
floor or goes out of bounds.

After each episode, the rewards that each agent received are added up and the score of the agent
that better performed is used. The challenge is considered solved when the average (over the last
100 episodes) of this score is at least 0.5.


---

# Dependencies

This project is a requirement from
the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
. The environment is provided by Udacity. It depends on the following packages:

- Python 3.6
- Numpy
- PyTorch
- Unity ML-Agents Beta v0.4

---

# Getting Started

## Linux (Debian-based)

- Install python3.6 (any version above is not compatible with the unity ml-agents version needed for
  this environment)

``` bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.6-full
```

- (Optional) Create a virtual environment for this project

```bash
cd <parent folder of venv>
python3.6 -m venv <name of the env>
source <path to venv>/bin/activate
```

- Install the python dependencies

``` bash
python3 -m pip install numpy torch
```

- Download the Unity
  ML-Agents [release file](https://github.com/Unity-Technologies/ml-agents/releases/tag/0.4.0b) for
  version Beta v0.4. Then, unzip it at folder of your choosing
- Build Unity ML-Agents

```bash
cd <path ml-agents>/python
python3 -m pip install .
```

- Clone this repository and download the environment created by Udacity and unzip it at the world
  folder

```bash
git clone https://github.com/jhonasiv/tennis-udacity
cd tennis-udacity
mkdir world
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip 
unzip Tennis_Linux.zip -d world
```

---

# Demo

After training the agent until a rolling average reward of 0.997 was reached for 100 episodes, this
is how it looks.

<p align="center">
  <img src="resources/tennis.gif" alt="animated"/>
  <p align="center">Agent trained with an average score of 0.997</p>
</p>

---

# Running the application

- Execute the main.py file
  ```bash
  python3 src/main.py
  ```
- For more information on the available command line arguments, use:
  ```bash
  python3 src/main.py --help
  ```
    - Some notable cli arguments:
        - `--eval`: runs the application in evaluation mode, skipping training step
        - `--buffer_size`: maximum size of the experience buffer.
        - `--a_lr`: learning rate for the actors
        - `--c_lr`: learning rate for the critics
