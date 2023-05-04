import gymnasium

gymnasium.envs.register(
    id='CartPoleReward-v1',
    entry_point='envs.cart-pole:CartPoleRewardEnv1',
)

gymnasium.envs.register(
    id='CartPoleObs-v1',
    entry_point='envs.cart-pole:CartPoleObsEnv1',
)

gymnasium.envs.register(
    id='CartPoleDist-v1',
    entry_point='envs.cart-pole:CartPoleDistEnv1',
)