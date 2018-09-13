import numpy as np

def score(board: np.ndarray, num=4):
    # For every 4*4 sub-board within the 6*7 board
    for row in range(board.shape[0]-num+1):
        for col in range(board.shape[1]-num+1):
            sub_board = board[row:row+num, col:col+num]
            
            # Check if any row, any column or any diagonal have 4 identical chess in a line.
            checklist = [list(sub_board.sum(axis=0)), list(sub_board.sum(axis=1)), [sub_board.trace()], [np.fliplr(sub_board).trace()]]
            
            # Return the winner
            for rule in checklist:
                if 4 in rule:
                    return 1
                elif -4 in rule:
                    return -1
    return 0


def place_chess(board: np.ndarray, action: int, role: int):
    location = 5-(np.fliplr(board.T)==0).argmax(axis=1)[action]
    board[location,action] = role
    return board