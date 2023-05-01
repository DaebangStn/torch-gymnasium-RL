import gymnasium

gymnasium.envs.register(
    id='CartPoleReward-v1',
    entry_point='envs.cart-pole:CartPoleRewardEnv1',
    max_episode_steps=1000,
)
