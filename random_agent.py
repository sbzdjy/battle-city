import retro
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

possible_actions = {
    # No Operation
    0: [0, 0, 0, 0, 0, 0, 0, 0, 0],
    # Up
    1: [0, 0, 0, 0, 1, 0, 0, 0, 0],
    # Down
    2: [0, 0, 0, 0, 0, 1, 0, 0, 0],
    # Left
    3: [0, 0, 0, 0, 0, 0, 1, 0, 0],
    # Right
    4: [0, 0, 0, 0, 0, 0, 0, 1, 0],
    # a
    5: [1, 0, 0, 0, 0, 0, 0, 0, 0],
}


def main():
    # retro.data.Integrations.add_custom_path(
    #     os.path.join(SCRIPT_DIR, "custom_integrations")
    # )

    env = retro.make("BattleCity-Nes") #, inttype=retro.data.Integrations.ALL)
    env.seed(0)
    obs = env.reset()

    while True:

        action = possible_actions[np.random.randint(len(possible_actions))]

        obs, rew, done, info = env.step(action)
        env.render()

        if done:
            print(f"Total reward: {rew}")
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
