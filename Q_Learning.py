import numpy as np
from RobocupSoccer import RobocupSoccer

def Q_learning(no_steps = int(1e5)):

    # Take action with epsilon-greedy r
    def generate_action(Q, state, epsilon):
        # epsilon-greey to take best action from action-value function
        if np.random.random() < epsilon:
            return np.random.choice([0,1,2,3,4], 1)[0]

        #return np.random.choice(np.flatnonzero(Q[state] == Q[state].max()))
        return np.random.choice(np.where(Q[state[0]][state[1]][state[2]] == max(Q[state[0]][state[1]][state[2]]))[0], 1)[0]

    np.random.seed(1)

    # discount factor
    gamma = 0.9

    # Define the epsilon and its decay for epsilon-greedy action selection
    epsilon = 1.0
    epsilon_min = 0.001
    epsilon_decay = 0.999995

    # epsilon_begin = 0.1
    # epsilon_end = 0
    # epsilon_periods = no_steps/2


    # learning rate
    alpha = 1.0
    alpha_min = 0.001
    alpha_decay = 0.999995
    #end_alpha = 0.001

    # store the step-error
    error_list = []

    # Q_tables of player A and player B
    # the state-action space is 8 (pos for player A) * 8 (pos for player B) * 2 (ball possession) * 5 (valid actions)
    Q_1 = np.zeros((8, 8, 2, 5))
    Q_2 = np.zeros((8, 8, 2, 5))

    # index for step
    i = 0

    while i < no_steps:
        env = RobocupSoccer()

        # map two players positions and ball possession into state presentation
        state = [env.pos[0][0] * 4 + env.pos[0][1], env.pos[1][0] * 4 + env.pos[1][1], env.ball]

        while True:
            if i % 1000 == 0:
                print('\rstep {}\t Time: {:.2f} \t Percentage: {:.2f}% \t Alpha: {:.3f}'.format(i, time.time() - start_time, i*100/no_steps, alpha), end="")

            # player A at sate S take action South before update
            # first index is player A's position index (0-7), 2 is frist row (0), 3rd column
            # second index is player B's position index (0-7), 1 is first row (0), 2nd column
            # third index is ball possession, according to graph, B has the ball
            # fourth index is action from player A, A is going south (2)
            before = Q_1[2][1][1][2]

            # eps-greedy to generate action
            actions = [generate_action(Q_1,state,epsilon), generate_action(Q_2,state,epsilon)]
            # get next state, reward and game termination flag
            state_prime, rewards, done = env.move(actions)

            i += 1

            # Q-learning for player A & B
            if done:
                Q_1[state[0]][state[1]][state[2]][actions[0]] = Q_1[state[0]][state[1]][state[2]][actions[0]] + alpha * (rewards[0] - Q_1[state[0]][state[1]][state[2]][actions[0]])

                Q_2[state[0]][state[1]][state[2]][actions[1]] = Q_2[state[0]][state[1]][state[2]][actions[1]] + alpha * (rewards[1] - Q_2[state[0]][state[1]][state[2]][actions[1]])

                # player A at state S take action South before update
                after = Q_1[2][1][1][2]
                error_list.append(abs(after-before))
                break

            else:
                Q_1[state[0]][state[1]][state[2]][actions[0]] = Q_1[state[0]][state[1]][state[2]][actions[0]] + alpha * (rewards[0] + gamma * max(Q_1[state_prime[0]][state_prime[1]][state_prime[2]]) - Q_1[state[0]][state[1]][state[2]][actions[0]])

                Q_2[state[0]][state[1]][state[2]][actions[1]] = Q_2[state[0]][state[1]][state[2]][actions[1]] + alpha * (rewards[1] + gamma * max(Q_2[state_prime[0]][state_prime[1]][state_prime[2]]) - Q_2[state[0]][state[1]][state[2]][actions[1]])
                state = state_prime

                # player A at state S take action South before update
                after = Q_1[2][1][1][2]
                error_list.append(abs(after-before))

            #decay epsilon and alpha
            epsilon *= epsilon_decay
            epsilon = max(epsilon_min, epsilon)

            alpha *= alpha_decay
            alpha = max(alpha_min, alpha)

    return error_list, Q_1, Q_2