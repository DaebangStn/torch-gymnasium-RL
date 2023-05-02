:star: RL agents and [Gymnasium](https://gymnasium.farama.org/) </br> environments in pyTorch :fire:
==============================
******************

RL agents (/agents)
--------------------
### 1. DQN 
* Saving the models 
> Path: /saved-models/{base_env_name}/{YY-mm-dd-HH-mm-ss-env_id}

Environment (/envs)
--------------------
### 1. CartPole
* Modified rewards to center the cart (CartPoleReward-v1)

Running (/runs)
--------------------
### 1. Training (train.py)
### 2. Testing (test.py)
* Show episode running by gif 

Logging (config.ini)
--------------------
* local console - WARNING
* file in /logs (TimedRotatingFileHandler) - DEBUG
* discord channel (discord_webhook) - None

:construction: Under Construction
==============================
******************
### 1. DQN [(torch tutorial)](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
--------------------
* Sample Benchmarking (MEAN durations vs Hyper parameters)
* Plot the reward and duration
* Show episode running by gif (add steps)
* Enlarge pole angle limit (CartPoleObs-v1)

### 2. Atari [(gymnasium doc)](https://gymnasium.farama.org/environments/atari/)
--------------------
* Implement Breakout
  * How to use frame skipping?
  * How to handle rgp array input?
