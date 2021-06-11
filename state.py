import chess
import numpy as np

VALUES = {'P': chess.PAWN, 'N': chess.KNIGHT, 'B': chess.BISHOP, 'R': chess.ROOK, 'Q': chess.QUEEN, 'K': chess.KING}

class State():
    def __init__(self, board):
        self.board = board

    def encode_board(self):
        x = np.zeros((6, 64), dtype=np.uint8)
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece != None:
                symbol = piece.symbol().upper()
                x[VALUES[symbol.upper()]-1][i] = 1 if symbol.isupper() else -1

        if not self.board.turn:
            x = -x
        return x.reshape(6, 8, 8)
