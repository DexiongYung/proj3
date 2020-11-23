import numpy as np
from RobocupSoccer import RobocupSoccer
import time


def Q_learning(no_steps, args):
    def generate_action(Q, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice([0, 1, 2, 3, 4], 1)[0]

        return np.random.choice(np.where(Q[state[0]][state[1]][state[2]] == max(Q[state[0]][state[1]][state[2]]))[0], 1)[0]

    np.random.seed(args.seed)

    gamma = args.gamma

    epsilon = args.eps
    epsilon_min = args.eps_min
    epsilon_decay = args.eps_decay

    alpha = args.alpha
    alpha_min = args.alpha_min
    alpha_decay = args.alpha_decay

    error_list = []

    Q_1 = np.zeros((8, 8, 2, 5))
    Q_2 = np.zeros((8, 8, 2, 5))

    i = 0
    start_time = time.time()

    while i < no_steps:
        env = RobocupSoccer()

        state = [env.pos[0][0] * 4 + env.pos[0][1],
                 env.pos[1][0] * 4 + env.pos[1][1], env.ball]

        while True:
            if i % 1000 == 0:
                print('\rstep {}\t Time: {:.2f} \t Percentage: {:.2f}% \t Alpha: {:.3f}'.format(
                    i, time.time() - start_time, i*100/no_steps, alpha), end="")

            before = Q_1[2][1][1][2]

            actions = [generate_action(
                Q_1, state, epsilon), generate_action(Q_2, state, epsilon)]
            state_prime, rewards, done = env.move(actions)

            i += 1

            if done:
                Q_1[state[0]][state[1]][state[2]][actions[0]] = Q_1[state[0]][state[1]][state[2]
                                                                                        ][actions[0]] + alpha * (rewards[0] - Q_1[state[0]][state[1]][state[2]][actions[0]])

                Q_2[state[0]][state[1]][state[2]][actions[1]] = Q_2[state[0]][state[1]][state[2]
                                                                                        ][actions[1]] + alpha * (rewards[1] - Q_2[state[0]][state[1]][state[2]][actions[1]])

                after = Q_1[2][1][1][2]
                error_list.append(abs(after-before))
                break

            else:
                Q_1[state[0]][state[1]][state[2]][actions[0]] = Q_1[state[0]][state[1]][state[2]][actions[0]] + alpha * \
                    (rewards[0] + gamma * max(Q_1[state_prime[0]][state_prime[1]]
                                              [state_prime[2]]) - Q_1[state[0]][state[1]][state[2]][actions[0]])

                Q_2[state[0]][state[1]][state[2]][actions[1]] = Q_2[state[0]][state[1]][state[2]][actions[1]] + alpha * \
                    (rewards[1] + gamma * max(Q_2[state_prime[0]][state_prime[1]]
                                              [state_prime[2]]) - Q_2[state[0]][state[1]][state[2]][actions[1]])
                state = state_prime

                after = Q_1[2][1][1][2]
                error_list.append(abs(after-before))

            epsilon *= epsilon_decay
            epsilon = max(epsilon_min, epsilon)

            alpha *= alpha_decay
            alpha = max(alpha_min, alpha)

    return error_list, Q_1, Q_2
