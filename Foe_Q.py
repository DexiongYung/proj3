from RobocupSoccer import RobocupSoccer
import numpy as np
import cvxpy as cp
import time
from Utilities import *
from Const import *


def Foe_Q(no_steps, args):
    def generate_action(pi, state, i):
        epsilon = epsilon_decay ** i
        if np.random.random() < epsilon:
            return np.random.choice([0, 1, 2, 3, 4], 1)[0]
        else:
            return np.random.choice([0, 1, 2, 3, 4], 1, p=pi[state[0]][state[1]][state[2]])[0]

    def max_min(Q, state):
        c = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        num_vars = len(c)
        x = cp.Variable(num_vars)
        objective = cp.Minimize(cp.sum(c @ x))

        G = np.array(np.append(np.append(np.ones(
            (5, 1)), -Q[state[0]][state[1]][state[2]], axis=1), np.append(np.zeros((5, 1)), -np.eye(5), axis=1), axis=0))
        h = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        A = np.array([[0.0], [1.0], [1.0], [1.0], [1.0], [1.0]])

        constraints = []
        constraints.append(G @ x <= h)
        constraints.append(A.T @ x == 1)

        prob = cp.Problem(objective, constraints)
        result = prob.solve()

        return np.abs(x.value[1:]).reshape((5,)) / sum(np.abs(x.value[1:])), np.array(x.value[0])

    gamma = args.gamma

    epsilon = args.eps
    epsilon_min = args.eps_min
    epsilon_decay = args.eps_decay

    alpha = args.alpha
    alpha_min = args.alpha_min
    alpha_decay = args.alpha_decay

    errors_list = []

    np.random.seed(1234)

    start_time = time.time()
    i = 0

    while i < no_steps:
        soccer = RobocupSoccer()
        state = [soccer.pos[0][0] * 4 + soccer.pos[0][1],
                 soccer.pos[1][0] * 4 + soccer.pos[1][1], soccer.ball]
        done = False
        while not done:
            if i % 1000 == 0:
                print('\rstep {}\t Time: {:.2f} \t Percentage: {:.2f}% \t Alpha: {:.3f}'.format(
                    i, time.time() - start_time, i*100/no_steps, alpha), end="")
            i += 1

            before = Q_1[2][1][1][4][2]

            actions = [generate_action(
                Pi_1, state, i), generate_action(Pi_2, state, i)]

            state_prime, rewards, done = soccer.move(actions)

            Q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]] = (
                1 - alpha) * Q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]] + alpha * (rewards[0] + gamma * V_1[state_prime[0]][state_prime[1]][state_prime[2]])

            pi, val = max_min(Q_1, state)
            Pi_1[state[0]][state[1]][state[2]] = pi
            V_1[state[0]][state[1]][state[2]] = val

            Q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]] = (
                1 - alpha) * Q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]] + alpha * (rewards[1] + gamma * V_2[state_prime[0]][state_prime[1]][state_prime[2]])

            pi, val = max_min(Q_2, state)
            Pi_2[state[0]][state[1]][state[2]] = pi
            V_2[state[0]][state[1]][state[2]] = val
            state = state_prime

            after = Q_1[2][1][1][4][2]
            errors_list.append(np.abs(after - before))

            alpha = alpha_decay ** i

    return errors_list, Q_1, Q_2, V_1, V_2, Pi_1, Pi_2
