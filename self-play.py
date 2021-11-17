import chess
import torch
import numpy as np

from chess import pgn

from mcts import MCTS
from model import Net
from state import State
from config import LABELS, N_LABELS, device
from util import move_lookup

DEBUG = 1
MOVE_LOOKUP = move_lookup(LABELS, N_LABELS)

def playWithMcts(p1, p2, board):
    
    while not board.is_game_over():

        if board.turn:
            mcts = MCTS(board, p1, 500)
            move, node = mcts.choose_move()
            move = move.uci()
            value = node.value()
            del mcts
        else:
            mcts = MCTS(board, p2, 500)
            move, node = mcts.choose_move()
            move = move.uci()
            value = node.value()
            del mcts

        board.push_san(move)
        if DEBUG:
            print(move, value, board.fen())


def playRaw(p1, p2, board):

    while not board.is_game_over():

        state = torch.from_numpy(State(board).encode_board()[np.newaxis]).to(device)

        if board.turn:
            with torch.no_grad():
                policy = p1(state)[0][0]

                # mask = torch.zeros((N_LABELS), device=device)
                # for move in board.legal_moves:
                #     mask[MOVE_LOOKUP[move.uci()]] = 1
                # policy *= mask

                maxIndex = torch.argmax(policy)
        else:
            with torch.no_grad():
                policy = p2(state)[0][0]
                maxIndex = torch.argmax(policy)

        move = list(MOVE_LOOKUP)[maxIndex]
        try:
            board.push_san(move)
        except ValueError:
            print(f'No legal move for {move} in {board.fen()} - {board.turn} to move')

        if DEBUG:
            print(move, board.fen())


if __name__ == "__main__":

    p1 = Net().to(device)
    p2 = Net().to(device)
    # p1.load_state_dict(torch.load('/mnt/Seagate/Code/chess-ai/model/model-adamw-epoch1.pth'))
    # p2.load_state_dict(torch.load('/mnt/Seagate/Code/chess-ai/model/model-adamw.pth'))
    p1.load_state_dict(torch.load('/mnt/Seagate/Code/chess-ai/model/model-adamw.pth'))
    p2.load_state_dict(torch.load('/mnt/Seagate/Code/chess-ai/model/model-adamw-epoch1.pth'))
    p1.eval()
    p2.eval()
    board = chess.Board()

    playWithMcts(p1, p2, board)
    # playRaw(p1, p2, board)
    print("Game Over", board.result())
    print(board.outcome())

    game = pgn.Game.from_board(board)
    game.headers["Event"] = "AdamW vs AdamW-Epoch1"
    game.headers["White"] = "AdamW"
    game.headers["Black"] = "AdamW-Epoch1"
    game.headers["Result"] = board.result()
    gameFile = open('/mnt/Seagate/Code/chess-ai/games/mcts-net-black.pgn', "w", encoding="utf-8")
    exporter = pgn.FileExporter(gameFile)
    game.accept(exporter)
