import chess
import numpy as np

from tqdm import tqdm
from chess import pgn
from glob import glob
from state import State
from train import DATA_DIR

RESULTS = {'0-1': -1, '1/2-1/2': 0, '1-0': 1}

def parse_dataset(data_dir):
    X, P, V = [], [], []

    # for file in tqdm(glob(data_dir + '/sample.pgn')):
    for file in tqdm(glob(data_dir + '/*.pgn')):
        games = open(file, encoding='utf-8')
        while (game := pgn.read_game(games)) is not None:
            # if len(list(game.mainline_moves())) == 0:
            #     continue
            board = chess.Board()
            result = RESULTS[game.headers['Result']]

            for move in game.mainline_moves():
                x, p = State(board).encode_board(), move.to_square
                X.append(x)
                P.append(p)
                V.append(result)
                if not board.turn:
                    X[-1] = -X[-1]
                    V[-1] = -V[-1]
                board.push(move)
        games.close()

    X, P, V = np.array(X, dtype=np.float32), np.array(P), np.array(V, dtype=np.float32)
    return X, P, V


if __name__ == '__main__':
    X, P, V = parse_dataset(DATA_DIR)
    np.savez(DATA_DIR + '/processed_data.npz', X, P, V)  
