from Foe_Q import Foe_Q
from CE_Q import CE_Q
from Friend_Q import Friend_Q
from Q_Learning import Q_learning
from Utilities import *
import argparse
import numpy as np

parser = argparse.ArgumentParser()
no_steps = 1e5
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_min', type=float, default=0.001)
parser.add_argument('--eps_decay', type=float, default=0.0001)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--alpha_min', type=float, default=0.0001)
parser.add_argument('--alpha_decay', type=float, default=0.0001)
parser.add_argument('--seed', type=int, default=1234)
args = parser.parse_args()


q_learning_errors, Q_1_qlearning, Q_2_qlearning = Q_learning(no_steps, args)
error_plot(np.array(q_learning_errors), 'Q-learner')

foe_q_errors, Q_1_foe, Q_2_foe, V_1_foe, V_2_foe, Pi_1_foe, Pi_2_foe = Foe_Q(
    no_steps, args)
error_plot(foe_q_errors, 'Foe-Q')

ce_q_errors, Q_1_ce, Q_2_ce, V_1_ce, V_2_ce, Pi_ce = CE_Q(no_steps, args)
error_plot(np.array(ce_q_errors), 'CE-Q')

friend_q_errors, Q_1_friend, Q_2_friend = Friend_Q(no_steps, args)
error_plot(np.array(friend_q_errors)[np.where(
    np.array(friend_q_errors) > 0)], 'Friend-Q')
