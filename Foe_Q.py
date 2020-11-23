from RobocupSoccer import RobocupSoccer
import numpy as np
import cvxpy as cp
import time
from Utilities import *

def Foe_Q(no_steps=int(1e5)):

    # Take action with epsilon-greedy
    def generate_action(pi, state, i):
        # epsilon-greey to take best action from action-value function
        # decay epsilon
        epsilon = epsilon_decay ** i
        if np.random.random() < epsilon:
            return np.random.choice([0, 1, 2, 3, 4], 1)[0]
        else:
            return np.random.choice([0, 1, 2, 3, 4], 1, p=pi[state[0]][state[1]][state[2]])[0]

    # same formulation as hw6
    # Q value is just like the reward matrix
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

    # discount factor
    gamma = 0.9

    # Define the epsilon and its decay for epsilon-greedy action selection
    epsilon = 1.0
    epsilon_min = 0.001
    epsilon_decay = 10**(np.log10(epsilon_min)/no_steps)
    # epsilon_min = 0.001
    # epsilon_decay = 0.999995

    # learning rate
    alpha = 1.0
    alpha_min = 0.001
    alpha_decay = 10**(np.log10(alpha_min)/no_steps)

    # Q_tables of player A and player B
    # the state-action space is 8 (pos for player A) * 8 (pos for player B) * 2 (ball possession) * 5 (valid actions for player A) * 5 (valid actions for player B)
    # initialization to 1 in order to break from zero
    Q_1 = np.ones((8, 8, 2, 5, 5)) * 1.0
    Q_2 = np.ones((8, 8, 2, 5, 5)) * 1.0

    # init policy for player 1 and player 2
    Pi_1 = np.ones((8, 8, 2, 5)) * 1/5
    Pi_2 = np.ones((8, 8, 2, 5)) * 1/5

    # value of states, only depends on pos of players and possession of ball
    V_1 = np.ones((8, 8, 2)) * 1.0
    V_2 = np.ones((8, 8, 2)) * 1.0

    # error list to store ERR
    errors_list = []

    # set seed
    np.random.seed(1234)

    # Loop for no_steps steps
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

            # player A at sate S take action South before update
            # first index is player A's position index (0-7), 2 is frist row (0), 3rd column
            # second index is player B's position index (0-7), 1 is first row (0), 2nd column
            # third index is ball possession, according to graph, B has the ball
            # fourth index is action from player B, B sticks
            # fifth index is action from player A, A goes south
            # rationale for putting player A's action as last index is for easy handling of max function (put the last dimention as player's action rather than opponent's action)
            before = Q_1[2][1][1][4][2]

            # eps-greedy to generate action
            actions = [generate_action(
                Pi_1, state, i), generate_action(Pi_2, state, i)]

            # get next state, reward and game termination flag
            state_prime, rewards, done = soccer.move(actions)

            # Q-learning update
            Q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]] = (
                1 - alpha) * Q_1[state[0]][state[1]][state[2]][actions[1]][actions[0]] + alpha * (rewards[0] + gamma * V_1[state_prime[0]][state_prime[1]][state_prime[2]])

            # use LP to solve maxmin
            pi, val = max_min(Q_1, state)
            Pi_1[state[0]][state[1]][state[2]] = pi
            V_1[state[0]][state[1]][state[2]] = val

            # Q-learning update
            Q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]] = (
                1 - alpha) * Q_2[state[0]][state[1]][state[2]][actions[0]][actions[1]] + alpha * (rewards[1] + gamma * V_2[state_prime[0]][state_prime[1]][state_prime[2]])

            # use LP to solve maxmin
            pi, val = max_min(Q_2, state)
            Pi_2[state[0]][state[1]][state[2]] = pi
            V_2[state[0]][state[1]][state[2]] = val
            state = state_prime

            # compute ERR
            after = Q_1[2][1][1][4][2]
            errors_list.append(np.abs(after - before))

            # decay learning rate
            alpha = alpha_decay ** i

    return errors_list, Q_1, Q_2, V_1, V_2, Pi_1, Pi_2


foe_q_errors, Q_1_foe, Q_2_foe, V_1_foe, V_2_foe, Pi_1_foe, Pi_2_foe = Foe_Q()
error_plot(np.array(foe_q_errors), 'Foe-Q')