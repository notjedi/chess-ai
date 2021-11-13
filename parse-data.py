import os
import torch
import chess

from tqdm import trange
from chess import pgn
from glob import glob
from torch import optim
from state import State
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Net, Loss, ChessDataset

DATA_DIR = '/mnt/Seagate/Code/chess-ai/data'
RESULTS = {'0-1': -1, '1/2-1/2': 0, '1-0': 1}
LIMIT = 1500000

def parse_dataset(file, net, opt, loss, writer, step):

    x, p, v = [], [], []

    games = open(file, encoding='utf-8')
    total_games = len(games.read().split('\n\n')) // 2
    games.close()
    games = open(file, encoding='utf-8')
    moves = 0

    for i in trange(total_games):

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
            moves += 1

        if (moves >= LIMIT or i == total_games-1):
            chess_dataset = ChessDataset(x, p, v)
            data_loader = DataLoader(chess_dataset, batch_size=128, shuffle=True, num_workers=6, drop_last=True)
            step = net.fit(data_loader, opt, loss, writer, step)
            torch.save(net.state_dict(), 'model/model.pth')
            moves = 0
            del data_loader
            del chess_dataset
            del x[:]
            del p[:]
            del v[:]

    games.close()
    return step


if __name__ == '__main__':

    net = Net()
    opt = optim.Adam(net.parameters())
    loss = Loss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.device(device)
    net.to(device)

    # summary(net, input_size=(6, 8, 8))
    net.train()
    net.load_state_dict(torch.load('model/model.pth'))
    writer = SummaryWriter()

    step = 0
    for file in glob(DATA_DIR + '/*.pgn'):
        print(file)
        step = parse_dataset(file, net, opt, loss, writer, step)
        os.system(f'mv {file} {file.replace("data", "processed")}')

    writer.close()
