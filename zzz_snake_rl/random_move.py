from zzz_snake_gym.env import ZzzSnakeEnv

if __name__ == '__main__':
    env = ZzzSnakeEnv()

    obs = env.reset()
    for i in range(10000):
        obs, reward, fail, over, info = env.step(env.action_space.sample())
        print(info)
        if fail or over:
            obs = env.reset()
