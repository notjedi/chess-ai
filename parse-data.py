import os
import chess
import numpy as np

from tqdm import trange
from chess import pgn
from glob import glob
from state import State
from train import DATA_DIR

RESULTS = {'0-1': -1, '1/2-1/2': 0, '1-0': 1}

def parse_dataset(data_dir):

    for file in glob(data_dir + '/*.pgn'):
        X, P, V = np.empty((0, 6, 8, 8), np.float32), np.empty(0), np.empty(0, np.float32)
        x, p, v = [], [], []

        games = open(file, encoding='utf-8')
        total_games = len(games.read().split('\n\n')) // 2
        games.close()
        games = open(file, encoding='utf-8')
        print(file)

        for i in trange(total_games):

            if (i % 1000 == 0 or i == total_games-1) and i != 0:
                X = np.concatenate((X, np.array(x)), axis=0)
                P = np.concatenate((P, np.array(p)), axis=0)
                V = np.concatenate((V, np.array(v)), axis=0)
                x, p, v = [], [], []

            game = pgn.read_game(games)
            board = chess.Board()
            result = RESULTS[game.headers['Result']]
            for move in game.mainline_moves():
                x.append(State(board).encode_board())
                p.append(move.to_square)
                v.append(result)

                if not board.turn:
                    v[-1] = -v[-1]
                board.push(move)

        np.savez(DATA_DIR + '/{}.npz'.format(os.path.basename(file)), X, P, V)
        games.close()


if __name__ == '__main__':
    parse_dataset(DATA_DIR)
