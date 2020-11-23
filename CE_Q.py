from RobocupSoccer import RobocupSoccer
from scipy.linalg import block_diag
from Utilities import *
import numpy as np
import cvxpy as cp
import time


def CE_Q(no_steps=int(10000)):

    # Take action with epsilon-greedy
    def take_action(Pi, state, i):
        # epsilon-greey to take best action from action-value function
        # decay epsilon
        epsilon = epsilon_decay ** i

        if np.random.random() < epsilon:
            index = np.random.choice(np.arange(25), 1)
            return np.array([index // 5, index % 5]).reshape(2)

        else:
            index = np.random.choice(
                np.arange(25), 1, p=Pi[state[0]][state[1]][state[2]].reshape(25))
            return np.array([index // 5, index % 5]).reshape(2)

    # using LP to solve correlated-equilibrium
    def solve_ce(Q_1, Q_2, state):
        # subset the condition for player A
        Q_states = Q_1[state[0]][state[1]][state[2]]
        s = block_diag(Q_states - Q_states[0, :], Q_states - Q_states[1, :], Q_states -
                       Q_states[2, :], Q_states - Q_states[3, :], Q_states - Q_states[4, :])
        row_index = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11,
                     13, 14, 15, 16, 17, 19, 20, 21, 22, 23]
        parameters_1 = s[row_index, :]

        # subset the condition for player B
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
        h = np.zeros(65) * 0.0
        A = np.ones((1, 25))

        objective = cp.Minimize(cp.sum(c @ x))

        constraints = []
        constraints.append(G @ x <= h)
        constraints.append(A @ x == 1)

        # error-handling mechanism
        try:
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
        except:
            print("error!!")
            prob = None
            val_1 = None
            val_2 = None

        return prob, val_1, val_2

    # discount rate
    gamma = 0.9
    epsilon_min = 0.001
    epsilon_decay = 10 ** (np.log10(epsilon_min)/no_steps)
    # epsilon_min = 0.001
    # epsilon_decay = 0.999995

    # learning rate
    alpha = 1.0
    alpha_min = 0.001
    alpha_decay = 10 ** (np.log10(alpha_min)/no_steps)

    # Q_tables of player A and player B
    # the state-action space is 8 (pos for player A) * 8 (pos for player B) * 2 (ball possession) * 5 (valid actions for player A) * 5 (valid actions for player B)
    Q_1 = np.ones((8, 8, 2, 5, 5)) * 1.0
    Q_2 = np.ones((8, 8, 2, 5, 5)) * 1.0

    # value of states, only depends on pos of players and possession of ball
    V_1 = np.ones((8, 8, 2)) * 1.0
    V_2 = np.ones((8, 8, 2)) * 1.0

    # shared joint policy
    Pi = np.ones((8, 8, 2, 5, 5)) * 1/25

    # error list to store ERR
    error_list = []

    # set seed
    np.random.seed(1234)

    start_time = time.time()
    i = 0
    while i < no_steps:
        soccer = RobocupSoccer()
        state = [soccer.pos[0][0] * 4 + soccer.pos[0][1],
                 soccer.pos[1][0] * 4 + soccer.pos[1][1], soccer.ball]
        done = 0
        j = 0
        while not done and j <= 100:
            if i % 1000 == 0:
                print('\rstep {}\t Time: {:.2f} \t Percentage: {:.2f}% \t Alpha: {:.3f}'.format(
                    i, time.time() - start_time, i*100/no_steps, alpha), end="")

            # udpate index
            i, j = i+1, j+1

            # we don't need place player B action space before player A
            # since we are no longer just selecting the max of player A
            before = Q_1[2][1][1][2][4]

            # eps-greedy to generate action
            actions = take_action(Pi, state, i)

            state_prime, rewards, done = soccer.move(actions)
            alpha = alpha_decay ** i

            # Q-learning update
            Q_1[state[0]][state[1]][state[2]][actions[0]][actions[1]] = (
                1 - alpha) * Q_1[state[0]][state[1]][state[2]][actions[0]][actions[1]] + alpha * (rewards[0] + gamma * V_1[state_prime[0]][state_prime[1]][state_prime[2]])

            # Q-learning update
            Q_2[state[0]][state[1]][state[2]][actions[1]][actions[0]] = (
                1 - alpha) * Q_2[state[0]][state[1]][state[2]][actions[1]][actions[0]] + alpha * (rewards[1] + gamma * V_2[state_prime[0]][state_prime[1]][state_prime[2]].T)
            prob, val_1, val_2 = solve_ce(Q_1, Q_2, state)

            # update only if not Null returned from the ce solver
            if prob is not None:
                Pi[state[0]][state[1]][state[2]] = prob
                V_1[state[0]][state[1]][state[2]] = val_1
                V_2[state[0]][state[1]][state[2]] = val_2
            state = state_prime

            # player A at state S take action South after update
            after = Q_1[2][1][1][2][4]
            # compute the error
            error_list.append(np.abs(after - before))

    return error_list, Q_1, Q_2, V_1, V_2, Pi

ce_q_errors, Q_1_ce, Q_2_ce, V_1_ce, V_2_ce, Pi_ce = CE_Q()
error_plot(np.array(ce_q_errors), 'CE-Q')