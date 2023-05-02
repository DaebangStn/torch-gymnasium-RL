:star: RL agents and [Gymnasium](https://gymnasium.farama.org/) </br> environments in pyTorch :fire:
==============================
******************

RL agents
--------------------
### 1. DQN
* Saving the models 
> Path: /saved-models/{base_env_name}/{YY-mm-dd-HH-mm-ss-env_id}

Environment
--------------------
### 1. CartPole
* Modified rewards to center the cart (CartPoleReward-v1)

Operating
--------------------
### 1. Training (/train.py)
### 2. Testing (/test.py)
### 3. Logging
* at the local console - WARNING
* at the file (TimedRotatingFileHandler) - DEBUG
* at the discord channel (discord_webhook) - ERROR

:construction: Under Construction
==============================
******************
### 1. DQN [(torch tutorial)](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
--------------------
* Sample Benchmarking (MEAN durations vs Hyper parameters)
* Plot the reward and duration
* Show episode running by gif 
* Enlarge pole angle limit (CartPoleObs-v1)

### 2. Atari [(gymnasium doc)](https://gymnasium.farama.org/environments/atari/)
--------------------
* Implement Breakout
  * How to use frame skipping?
  * How to handle rgp array input?
