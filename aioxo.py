# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 01:12:35 2015

@author: mooreh
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping


# Reads current state of the board
def EvaluateBoard(board):
    """Returns one of (-1, 0, 1) that correspond to (win for X, draw, win for O)
    """
    # Parameter is the board matrix, called "board" throughout code
    rows = board.sum(axis=1).flatten()
    # Variable "rows" defined as sum of values in the 1st axis (horizontal)
    # .flatten() returns the output of the matrix in one dimension
    cols = board.sum(axis=0)
    # Variable "cols" defined as sum in 0th axis
    diag1 = board.diagonal()
    # .diagonal() returns the diagonal of a matrix
    diag2 = (np.fliplr(board)).diagonal()
    # The fliplr function from the numpy library flips an array in the left/right (l/r) direction
    # .diagonal() now returns a different diagonal from diag1

    # Calculate Max and Min values from all four
    max_val_list = [rows.max(), cols.max(), diag1.max(), diag2.max()]
    min_val_list = [rows.min(), cols.min(), diag1.min(), diag2.min()]

    if 3 in max_val_list:
        return 1
    # Players' pieces on the board are indicated by 1's and -1's as supposed to the symbols used in regular games ("X"/"O")
    # If one of the four sums above (rows, cols, diag1 or diag2) have a value of 3 (i.e. three 1's in a row), a 1 is returned

    if -3 in min_val_list:
        return -1
    # Similarly, if a sum of -3 is counted (three -1's in a row, indicating an opponent's win), a -1 is returned

    return 0
    # This last line returns a zero, and only runs when there is no 3 in a row.
    # This will occur during the game while in play and there is no winner yet, and also when an un-winnable/drawn board has happened


# This initializes the neural network with a neuron for each square in the board (defaulting at 9) plus an input and output
def InitNN(BOARD_SIZE):
    """Returns a matrix of randomly selected weights
    For board size 3, maximum weight will be 11
    """
    # The board_size value takes the length of one side of the board matrix
    # This is required to correctly setup the appropriate weights for the neurons
    weights = (neurons * BOARD_SIZE**2) + (2 * neurons)
    # Weights is simply calculated through multiplying and adding the values for neurons and the board_size above
    return np.matrix(np.random.randn(1, weights))
    # Creates the initial matrix, with random values upto and including the weights


# Runs the neural network aginast the current board
def RunNN(nn, board):
    """Returns value associated with the board (used in determining best board position)
    """
    w = nn[:, 0]
    # The W variable is the first value in this variable - the input value for the network
    x = nn[:, 1]
    # The X variable is the second value in this variable - the bias value
    z = nn[:, 2:]
    # The Z variable is every other value - the neuron values
    return np.dot(w, np.tanh(np.dot(z, board.flatten().getT()) + x))
    # The function is the activation function for the network. This takes the dot product of the transposed matrix of tahn'ed results and the input (i.e. the board state)


# Runs the AI by evaluating against every square in the board (and finds the 'best' move)
def RunAI(ai, board, side):
    """Returns coordinates for best move on the board
    """
    mymax = -10000000000
    for i in range(3):
        for j in range(3):
            b = board.copy()
            # The board is copied
            b[i, j] = side
            # Variable side is the (j, i) position of the game board
            value = RunNN(ai, b)
            # The RunNN fuction takes the copied board and the current state of the AI
            if value > mymax and board[i, j] == 0:
                # This while loop continues to run whilst the next position in the board either returns a worse value from the Neural Network, and has actually been played on
                mymax = value
                # The value of this position is updated
                q = i  # Best y
                r = j  # Best x
                # The co-ords of the best position are updated
    return q, r


# A piece is placed on the board
def Move(ai, board, side):
    """Returns updated board with the best move for `side` (Random player or AI)
    """
    # RunAI function is run to pick which position to play
    i, j = RunAI(ai, board, side)
    # Board position from function is placed in the matrix
    board[i, j] = side
    # Returns the updated game board
    return board


# Returns boolean of whether there is still a free space on the board
def IsMovePossible(board):
    """Returns True if no space is left on board, else False
    """
    return np.count_nonzero(board) < 9


# Deals with the whole process of playing an OXO game
def NNvsRandomPlayer(ai):
    """Returns (-1, 0, 1) according to the winner
    """
    # Starting player picked
    side = np.random.choice([1, -1])
    # Board initialised
    board = np.matrix('0 0 0 ;0 0 0;0 0 0')
    # Initialises won variable for while loop
    won = 0
    while won == 0 and IsMovePossible(board):
        # If player is NNAI, run function to choose move
        if side == 1:
            board = Move(ai, board, side)
        else:
            # Process to choose random position that isn't already played
            x = np.random.choice(np.where(board == 0)[0].getA().flatten())
            y = np.random.choice(np.where(board == 0)[1].getA().flatten())
            board[x, y] = side
        won = EvaluateBoard(board)
        side = side * -1
    return won


#: Returns result of using `basinhopping` minimizing algorithm on `optthis` function
def TrainAI():
    """Result contains `x`: point where min is achieved and `fun`, the min value of function at that point
    """
    def optthis(x):
        ai = np.matrix(x)
        thesum1 = []
        for j in range(10):
            thesum = 0
            for i in range(10):
                ev = NNvsRandomPlayer(ai)
                thesum += ev
            thesum = thesum / 10.0
            thesum1.append(thesum)
        ev = np.median(thesum1)
        ev = -1 * ev
        return ev

    ai2 = InitNN(BOARD_SIZE)
    optresult = basinhopping(optthis, ai2, disp=True)
    return optresult


BOARD_SIZE = 3
neurons = 1
board = np.matrix('0 0 0;0 0 0;0 0 0')

labels = ('X', 'O', 'Draw')
colors=('red', 'blue', 'gold')

print "Training NN"
trained = TrainAI()
print trained
ai = np.matrix(trained.x)

print "BOARD with new ai"
sizes = [0, 0, 0]
for loop in range(10000):
    result = NNvsRandomPlayer(ai)
    if result == -1:
        sizes[0] += 1
    elif result == 1:
        sizes[1] += 1
    else:
        sizes[2] += 1
plt.pie(sizes,
        explode=None,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        shadow=True)
plt.axis('equal')
plt.show()
print sizes
