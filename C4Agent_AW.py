import numpy as np
import math
from functools import lru_cache, wraps
from re import search, findall

def np_cache(*args, **kwargs):
    ''' LRU cache implementation for functions whose FIRST parameter is a numpy array
        forked from: https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75 '''

    def decorator(function):
        @wraps(function)
        def wrapper(np_array, *args, **kwargs):
            hashable_array = tuple(map(tuple, np_array))
            return cached_wrapper(hashable_array, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(hashable_array, *args, **kwargs):
            array = np.array(hashable_array)
            return function(array, *args, **kwargs)

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear
        return wrapper
    return decorator

# Helper functions

def empty_board(shape=(6, 7)):
    return np.full(shape=shape, fill_value=' ')

def drop(board, piece, column, inplace=True):
    board_ = np.array(board, copy=not inplace)
    for row in reversed(board_):
        if row[column] == ' ':
            row[column] = piece
            return board_

def utility(board, player, length=4):
    """Determines utility of board for player if terminal"""
    state = terminal(board, length)
    if state == player: return 1
    if state == 'draw': return 0
    if state:           return -1

def terminal(board, length=4, last_move=None):
    """Returns piece of the player that has won.
    Returns 'draw' if draw, otherwise board is not
    in terminal state and returns None."""

    cols = board.shape[1]

    chars = [last_move]
    if last_move is None:
        chars = np.unique(board).tolist()
        if ' ' in chars: chars.remove(' ')

    # unravel board into string
    board_str = '|'.join(str(row.tobytes().replace(b'\x00', b'').decode("utf-8")) for row in board)
    
    # check for four in a line
    for char in chars:
        if search(char * length, board_str):
            # horizontal
            return char
        if search(char + ('.' * cols + char) * (length - 1), board_str):
            # vertical
            return char
        if search(char + ('.' * (cols + 1) + char) * (length - 1), board_str):
            # diagonal top-left to bottom-right
            return char
        if search(char + ('.' * (cols - 1) + char) * (length - 1), board_str):
            # diagonal top-right to bottom-left
            return char
    
    # if no matches and all columns are filled,
    # then draw has been reached
    if ' ' not in board:
        return 'draw'

    # if none of previous conditions are satisfied,
    # board is not in terminal state.
    return None

line_counts = {
    (0,0):4, (0,1):6, (0,2):7, (0,3):8,
    (1,0):2, (1,1):4, (1,2):6, (1,3):8,
    (2,0):0, (2,1):2, (2,2):5, (2,3):8
}

def line_count(row, col):
    return line_counts[(-abs(2*row - 5) + 5)//2, abs(col - 3)]

def actions(board):
    spaces = {}
    for col in range(board.shape[1]):
        for row in reversed(range(board.shape[0])):
            if board[row, col] == ' ':
                spaces[col] = row
                break
    moves = [*spaces]
    return sorted(moves, key=lambda c: line_count(spaces[c], c))

@lru_cache
def other(player):
    return 'o' if player == 'x' else 'x'

# Heuristic functions

@lru_cache
def heuristic_pattern(char, orientation, cols, length, space):
    o = orientation % 180
    if o == 0:    o = cols
    elif o == 45: o = 1
    elif o == 90: o = 0
    elif o == 135:o = -1
    pattern = '(?=' + (char + '.' * (cols - o)) * space + ' ' + ('.' * (cols - o) + char) * (length - space - 1) + ')'
    return pattern

@np_cache(maxsize=2**19)
def evaluate(board, player='x', length=4):
    """Heuristic for the utility of the board. Returns the score.

    If a player has already won, a scaled utility is returned.
    Otherwise, returns the number of lines that are one empty
    space away from winning the game for the player subtracted by
    similar lines for the opposing player."""
    u = utility(board, player)
    if u is not None:
        return u * board.size * 4, True

    # unravel board into string
    board_str = '|'.join(str(row.tobytes().replace(b'\x00', b'').decode("utf-8")) for row in board)
    cols = board.shape[1]

    score = 0
    for i in range(length):
        for o in range(0, 180, 45):
            score += len(findall(heuristic_pattern(player, o, cols, length, i), board_str))
            score -= len(findall(heuristic_pattern(other(player), o, cols, length, i), board_str))
    return score, False

# Heuristic Alpha-Beta Search algorithm functions

def good_move(board, player, alpha=-math.inf, beta=+math.inf, depth=0, cutoff=8):
    """Returns heuristically best move of player and its value"""
    value, terminal = evaluate(board, player)
    if depth > cutoff or terminal:
        if terminal: alpha, beta = value, value
        return None, value

    move, value = None, -math.inf
    for a in actions(board):
        a_, v_ = bad_move(drop(board, player, a, inplace=False), player, alpha, beta, depth + 1, cutoff)
        if v_ > value:
            move, value = a, v_
            alpha = max(alpha, value)
        if value >= beta: return move, value
    return move, value

def bad_move(board, player, alpha=-math.inf, beta=+math.inf, depth=0, cutoff=8):
    """Returns heuristically best move of player and its value"""
    value, terminal = evaluate(board, player)
    if terminal:
        alpha, beta = value, value
        return None, value

    move, value = None, +math.inf
    for a in actions(board):
        a_, v_ = good_move(drop(board, other(player), a, inplace=False), player, alpha, beta, depth + 1, cutoff)
        if v_ < value:
            move, value = a, v_
            beta = min(beta, value)
        if value <= alpha: return move, value
    return move, value

# Monte Carlo Search functions

def playout(board, player, move):
    state = drop(board, player, move, inplace=False)
    current = other(player)
    u = utility(board, player)
    while u is None:
        moves = actions(state)
        drop(state, current, moves[int(np.random.exponential(10) % len(moves))])
        current = other(current)
        u = utility(state, player)
    return u

def simulate(board, player='x', N=10000):
    """Returns the action with largest average utility as
    determined by a pure Monte Carlo search."""
    C = math.sqrt(2)
    moves = actions(board)
    u = dict.fromkeys(moves, 0)
    n = dict.fromkeys(moves, 0)
    n_parent = 0
    UCB1 = dict.fromkeys(moves, +math.inf)
    for i in range(N):
        move = max(UCB1, key=UCB1.get)
        p = playout(board, player, move)
        u[move] += p
        n[move] += 1
        n_parent += 1
        for move in moves:
            if n[move] > 0:
                UCB1[move] = u[move]/n[move] + C * math.sqrt(math.log(n_parent) / n[move])
    return max(n, key=n.get)

# Connect Four Agent

#import time

def agent(board, player='x'):
    #start = time.time()
    move_count = np.count_nonzero(board == player)
    move = board.shape[1] // 2
    if move_count != 0 and move_count < 8:
        move, _ = good_move(board, player)
    else:
        move = simulate(board, player)
    drop(board, player, move)
    #print(time.time() - start)
    return move