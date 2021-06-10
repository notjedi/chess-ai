import chess
import numpy as np

VALUES = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6}

class State():
    def __init__(self, board, move=None):
        self.board = board
        self.move = move

    def encode_board(self):
        x = np.zeros((6, 64), dtype=np.uint8)
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece != None:
                symbol = piece.symbol().upper()
                x[VALUES[symbol.upper()]-1][i] = 1 if symbol.isupper() else -1

        return x.reshape(6, 8, 8)

    def encode_move(self):
        move = self.move.__str__()
        try:
            return chess.parse_square(move[-2:])
        except ValueError:
            return chess.parse_square(move[2:4])
