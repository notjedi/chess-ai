import os
import torch
import chess

from tqdm import trange
from chess import pgn
from glob import glob
from torch import optim
from state import State
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Net, Loss, ChessDataset
from config import N_LABELS, LABELS, device
from util import move_lookup

MOVE_LOOKUP = move_lookup(LABELS, N_LABELS)
DATA_DIR = '/mnt/Seagate/Code/chess-ai/data'
RESULTS = {'0-1': -1, '1/2-1/2': 0, '1-0': 1}
LIMIT = 2000000

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
            p.append(MOVE_LOOKUP[move.uci()])
            v.append(result)

            if not board.turn:
                v[-1] = -v[-1]
            board.push(move)
            moves += 1

        if (moves >= LIMIT or i == total_games-1):
            chess_dataset = ChessDataset(x, p, v)
            data_loader = DataLoader(chess_dataset, batch_size=512, shuffle=True, drop_last=True)
            del x[:]
            del p[:]
            del v[:]
            step = net.fit(data_loader, opt, loss, writer, step)
            torch.save(net.state_dict(), 'model/model.pth')
            moves = 0
            del data_loader
            del chess_dataset

    games.close()
    return step


if __name__ == '__main__':

    # TODO: gradient accumulation?

    # TODO: doesn't know how to checkmate or even more generally what moves to make nearing the endgame
    # (repetition in moves, doesn't promote even when it's 1 move away, misses easy checkmate opportunities)
    # should i train on low elo games(cause gm games usually end in a draw)? or is it a search problem in mcts?

    # Initial Loss is 8 which is really expected because one would expect the intitial loss to be 7.58,
    # because the probablity would be somewhat near 1/1968. -ln(1/1968) = 7.58.

    torch.manual_seed(1337)
    net = Net()
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
    # weight decay(of 0.1) abosolutely destroys the model idk why
    opt = optim.AdamW(net.parameters(), lr=0.001)
    loss = Loss()

    net.to(device)

    # summary(net, input_size=(6, 8, 8))
    net.train()
    # net.load_state_dict(torch.load('model/model.pth'))
    writer = SummaryWriter()

    step = 0
    for file in glob(DATA_DIR + '/*.pgn'):
        print(file)
        step = parse_dataset(file, net, opt, loss, writer, step)
        os.system(f'mv {file} {file.replace("data", "processed")}')

    writer.close()
