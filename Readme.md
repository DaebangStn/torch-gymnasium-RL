:star: RL agents and [Gymnasium](https://gymnasium.farama.org/) </br> environments in pyTorch :fire:
==============================
******************

RL agents (/agents)
--------------------
### 1. DQN 
* Saving the models 
> Path: /saved-models/{base_env_name}/{YY-mm-dd-HH-mm-ss-env_id}

Environments (/envs)
--------------------
### 1. CartPole
* Modified rewards to center the cart (CartPoleReward-v1)
* Enlarge pole angle limit (CartPoleObs-v1)
* Random disturbance (CartPoleDisturbance-v1)
* Render with custom text (render_with_info)

### 2. Atari [(gymnasium doc)](https://gymnasium.farama.org/environments/atari/)
--------------------
* Test training available

Running (/runs)
--------------------
### 1. Training (train.py)
### 2. Testing (test.py)
* Show episode running by gif 
### 3. Benchmarking (benchmark.py)
* Simple Benchmarking (Hyper parameters)

Plotting (/plots)
--------------------
* Plot the reward and steps

Logging (config.ini)
--------------------
* local console - WARNING
* file in /logs (TimedRotatingFileHandler) - DEBUG
* discord channel (discord_webhook) - None



:construction: Under Construction
==============================
******************
* Restructure the whole project to Wrapper based

Logging (config.ini)
--------------------
* telegram bot - not working, seems to there is an error on handler emit method
