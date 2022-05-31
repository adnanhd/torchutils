from gym.envs.registration import register

register(
        id='loss-v0',
        entry_point='gym_loss.envs:LossEnv',
        )
