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
from config import N_LABELS, LABELS

# MOVE_LOOKUP = {chess.Move.from_uci(move): i for move, i in zip(LABELS, range(len(LABELS)))}
MOVE_LOOKUP = {move: i for move, i in zip(LABELS, range(N_LABELS))}
DATA_DIR = '/mnt/Seagate/Code/chess-ai/data'
RESULTS = {'0-1': -1, '1/2-1/2': 0, '1-0': 1}
LIMIT = 1500000
MINELO = 1900

def parse_dataset(file, net, opt, loss, writer, step, device):

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
            chess_dataset = ChessDataset(x, p, v, device)
            data_loader = DataLoader(chess_dataset, batch_size=128, shuffle=True, num_workers=6, drop_last=True, pin_memory=True)
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

    # TODO: use amp scalar
    # TODO: schedule learning rate
    # TODO: refresh NLL and cross-entropy loss
    # TODO: grad check intuitively during the initial stages of training
    # TODO: remove bias from CNN before BN

    # TODO: test with AdamW?
    # TODO: gradient accumulation?
    # TODO: turn on cudNN benchmarking?
    # TODO: self play module to eval the model?
    # TODO: doesn't know how to checkmate or even more generally what moves to make nearing
    # the endgame (repetition in moves, doesn't promote even when it's 1 move away, misses easy checkmate opportunities) 
    net = Net()
    opt = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.1)
    loss = Loss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.device(device)
    net.to(device)

    # summary(net, input_size=(6, 8, 8))
    net.train()
    # net.load_state_dict(torch.load('model/model.pth'))
    writer = SummaryWriter()

    step = 0
    for file in glob(DATA_DIR + '/*.pgn'):
        print(file)
        step = parse_dataset(file, net, opt, loss, writer, step, device)
        os.system(f'mv {file} {file.replace("data", "processed")}')

    writer.close()
