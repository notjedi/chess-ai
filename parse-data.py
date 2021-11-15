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
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from model import Net, Loss, ChessDataset
from config import N_LABELS, LABELS, device
from util import move_lookup

MOVE_LOOKUP = move_lookup(LABELS, N_LABELS)
DATA_DIR = '/mnt/Seagate/Code/chess-ai/data'
RESULTS = {'0-1': -1, '1/2-1/2': 0, '1-0': 1}
LIMIT = 1500000
MINELO = 1900

def parse_dataset(file, net, opt, scheduler, loss, writer, step):

    x, p, v = [], [], []

    games = open(file, encoding='utf-8')
    total_games = len(games.read().split('\n\n')) // 2
    games.close()
    games = open(file, encoding='utf-8')
    moves = 0

    for i in trange(total_games):

        game = pgn.read_game(games)
        board = chess.Board()
        whiteElo = int(game.headers['WhiteElo'])
        blackElo = int(game.headers['BlackElo'] )
        if (whiteElo < MINELO or blackElo < MINELO):
            continue

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
            step = net.fit(data_loader, opt, scheduler, loss, writer, step)
            torch.save(net.state_dict(), 'model/model.pth')
            moves = 0
            del data_loader
            del chess_dataset

    games.close()
    return step


if __name__ == '__main__':

    # TODO: test with AdamW?
    # TODO: gradient accumulation?
    # TODO: explore other lr schedulers

    # TODO: self play module to eval the model?
    # TODO: doesn't know how to checkmate or even more generally what moves to make nearing the endgame
    # (repetition in moves, doesn't promote even when it's 1 move away, misses easy checkmate opportunities)
    # should i train on low elo games(cause gm games usually end in a draw)? or is it a search problem in mcts?

    # Initial Loss is 8 which is really expected because one would expect the intitial loss to be 7.58,
    # because the probablity would be somewhat near 1/1968. -ln(1/1968) = 7.58.

    torch.backends.cudnn.benchmark = True
    net = Net()
    opt = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.1)
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
    scheduler = StepLR(opt, step_size=20000, gamma=0.1)
    loss = Loss()

    torch.device(device)
    net.to(device)

    # summary(net, input_size=(6, 8, 8))
    net.train()
    # net.load_state_dict(torch.load('model/model.pth'))
    writer = SummaryWriter()
    model = torch.nn.DataParallel(net).to(device)

    step = 0
    for file in glob(DATA_DIR + '/*.pgn'):
        print(file)
        step = parse_dataset(file, net, opt, scheduler, loss, writer, step)
        os.system(f'mv {file} {file.replace("data", "processed")}')

    writer.close()
