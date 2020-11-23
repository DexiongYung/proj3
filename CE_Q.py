from RobocupSoccer import RobocupSoccer
from scipy.linalg import block_diag
from Utilities import *
from Const import *
import numpy as np
import cvxpy as cp
import time


def CE_Q(no_steps, args):
    def take_action(Pi, state, i):
        epsilon = epsilon_decay ** i

        if np.random.random() < epsilon:
            index = np.random.choice(np.arange(25), 1)
            return np.array([index // 5, index % 5]).reshape(2)

        else:
            index = np.random.choice(
                np.arange(25), 1, p=Pi[state[0]][state[1]][state[2]].reshape(25))
            return np.array([index // 5, index % 5]).reshape(2)

    def solve(Q_1, Q_2, state):
        Q_states = Q_1[state[0]][state[1]][state[2]]
        s = block_diag(Q_states - Q_states[0, :], Q_states - Q_states[1, :], Q_states -
                       Q_states[2, :], Q_states - Q_states[3, :], Q_states - Q_states[4, :])
        row_index = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11,
                     13, 14, 15, 16, 17, 19, 20, 21, 22, 23]
        parameters_1 = s[row_index, :]

        Q_states = Q_2[state[0]][state[1]][state[2]]
        s = block_diag(Q_states - Q_states[0, :], Q_states - Q_states[1, :], Q_states -
                       Q_states[2, :], Q_states - Q_states[3, :], Q_states - Q_states[4, :])
        col_index = [0, 5, 10, 15, 20, 1, 6, 11, 16, 21, 2,
                     7, 12, 17, 22, 3, 8, 13, 18, 23, 4, 9, 14, 19, 24]
        parameters_2 = s[row_index, :][:, col_index]

        c = np.array((Q_1[state[0]][state[1]][state[2]] +
                      Q_2[state[0]][state[1]][state[2]].T).reshape(25))
        num_vars = len(c)
        x = cp.Variable(num_vars)
        G = np.append(np.append(parameters_1, parameters_2,
                                axis=0), -np.eye(25), axis=0)
        h = np.zeros(65).astype(float)
        A = np.ones((1, 25))

        objective = cp.Minimize(cp.sum(c @ x))

        constraints = []
        constraints.append(G @ x <= h)
        constraints.append(A @ x == 1)

        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        if x.value is not None:
            prob = np.abs(np.array(x.value).reshape(
                (5, 5))) / sum(np.abs(x.value))
            val_1 = np.sum(prob * Q_1[state[0]][state[1]][state[2]])
            val_2 = np.sum(prob * Q_2[state[0]][state[1]][state[2]].T)
        else:
            prob = None
            val_1 = None
            val_2 = None

        return prob, val_1, val_2

    gamma = args.gamma

    alpha = args.alpha
    alpha_min = args.alpha_min
    alpha_decay = args.alpha_decay

    epsilon = args.eps
    epsilon_decay = args.eps_decay

    Pi = np.ones((8, 8, 2, 5, 5)) * 1/25

    errors = []

    np.random.seed(args.seed)

    start_time = time.time()
    i = 0
    while i < no_steps:
        env = RobocupSoccer()
        state = init(env)
        done = False
        j = 0
        while not done and j <= 100:
            if i % args.print == 0:
                print('\rstep {}\t Time: {:.2f} \t Percentage: {:.2f}% \t Alpha: {:.3f}'.format(
                    i, time.time() - start_time, i*100/no_steps, alpha), end="")

            Q_t = Q_1[2][1][1][2][4]

            actions = take_action(Pi, state, i)

            state_prime, rewards, done = env.move(actions)
            alpha = alpha_decay ** i

            Q_1[state[0]][state[1]][state[2]][actions[0]][actions[1]] = (
                1 - alpha) * Q_1[state[0]][state[1]][state[2]][actions[0]][actions[1]] + alpha * (rewards[0] + gamma * V_1[state_prime[0]][state_prime[1]][state_prime[2]])

            Q_2[state[0]][state[1]][state[2]][actions[1]][actions[0]] = (
                1 - alpha) * Q_2[state[0]][state[1]][state[2]][actions[1]][actions[0]] + alpha * (rewards[1] + gamma * V_2[state_prime[0]][state_prime[1]][state_prime[2]].T)
            prob, val_1, val_2 = solve(Q_1, Q_2, state)

            if prob is not None:
                Pi[state[0]][state[1]][state[2]] = prob
                V_1[state[0]][state[1]][state[2]] = val_1
                V_2[state[0]][state[1]][state[2]] = val_2
            
            state = state_prime
            Q_tp1 = Q_1[2][1][1][2][4]
            errors.append(np.abs(Q_tp1 - Q_t))
            i, j = i+1, j+1

    return errors, Q_1, Q_2, V_1, V_2, Pi
