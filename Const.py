import numpy as np
Q_1 = np.ones((8, 8, 2, 5, 5)) * 1.0
Q_2 = np.ones((8, 8, 2, 5, 5)) * 1.0

# init policy for player 1 and player 2
Pi_1 = np.ones((8, 8, 2, 5)) * 1/5
Pi_2 = np.ones((8, 8, 2, 5)) * 1/5

# value of states, only depends on pos of players and possession of ball
V_1 = np.ones((8, 8, 2)) * 1.0
V_2 = np.ones((8, 8, 2)) * 1.0


def init(env):
    return [env.pos[0][0] * 4 + env.pos[0][1],
            env.pos[1][0] * 4 + env.pos[1][1], env.ball]
