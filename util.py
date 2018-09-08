import numpy as np

def score(board: np.ndarray, num=4):
    for row in range(board.shape[0]-num+1):
        for col in range(board.shape[1]-num+1):
            sub_board = board[row:row+num, col:col+num]
            checklist = [list(sub_board.sum(axis=0)), list(sub_board.sum(axis=1)), [sub_board.trace()], [np.fliplr(sub_board).trace()]]
            for rule in checklist:
                if 1 in rule:
                    return 1
                elif -1 in rule:
                    return -1
    return 0