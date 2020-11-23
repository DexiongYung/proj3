import numpy as np

LEGAL_ACTIONS = [[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]]


class RobocupSoccer:
    def __init__(self):
        self.pos = [np.array([0, 2]), np.array([0, 1])]
        self.ball = 1
        self.goal = [0, 3]

    def move(self, actions):
        first = np.random.randint(2, size=1)[0]
        second = 1 - first
        new_pos = self.pos.copy()
        scores = np.array([0, 0])
        is_done = False

        if actions[0] not in range(0, 5) or actions[1] not in range(0, 5):
            invalid_player = 1 if actions[0] not in range(0, 5) else 0
            bad_action = actions[0] if actions[0] not in range(
                0, 5) else actions[1]
            raise Exception(
                f'Player {invalid_player} has made an invalid action: {bad_action}')
        else:
            new_pos[first] = self.pos[first] + \
                LEGAL_ACTIONS[actions[first]]

            if (new_pos[first] == self.pos[second]).all():
                if self.ball == first:
                    self.ball = second

            elif new_pos[first][0] in range(0, 2) and new_pos[first][1] in range(0, 4):
                self.pos[first] = new_pos[first]
                if self.pos[first][1] == self.goal[first] and self.ball == first:
                    scores = ([1, -1][first]) * np.array([100, -100])
                    is_done = True
                    return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.ball], scores, is_done
                elif self.pos[first][1] == self.goal[second] and self.ball == first:
                    scores = ([1, -1][first]) * np.array([-100, 100])
                    is_done = True
                    return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.ball], scores, is_done
            new_pos[second] = self.pos[second] + \
                LEGAL_ACTIONS[actions[second]]
            if (new_pos[second] == self.pos[first]).all():
                if self.ball == second:
                    self.ball = first
            elif new_pos[second][0] in range(0, 2) and new_pos[second][1] in range(0, 4):
                self.pos[second] = new_pos[second]
                if self.pos[second][1] == self.goal[second] and self.ball == second:
                    scores = ([1, -1][second]) * np.array([100, -100])
                    is_done = True
                    return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.ball], scores, is_done
                elif self.pos[second][1] == self.goal[first] and self.ball == second:
                    scores = np.array([-100, 100]) * [1, -1][second]
                    is_done = True
                    return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.ball], scores, is_done

        return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.ball], scores, is_done
