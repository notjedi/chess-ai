import torch
import chess

from torch import nn
from chess import pgn
from glob import glob
from torchsummary import summary
from torchvision import transforms
from collections import OrderedDict
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

DATA_DIR = '/mnt/Seagate/Code/chess-ai/data'

class ChessDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        self.games = []
        self.results = {'0-1': -1, '1/2-1/2': 0, '1-0': 1}
        self.values = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6}
        # self.values = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        #                'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6, None: 0}

        for file in glob(data_dir + '/*.pgn'):
            with open(file, encoding='utf-8') as f:
                while (game := pgn.read_game(f)) != None:
                    self.games.append(game)

        if transform is not None:
            self.games = transform(self.games)

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        return self.encode(self.games[idx])

    def state(self, board):
        x = torch.zeros((6, 64), dtype=torch.uint8)
        for i in range(64):
            piece = board.piece_at(i)
            if piece != None:
                symbol = piece.symbol().upper
                x[self.values[symbol.upper()]][i] = 1 if symbol.is_upper() else -1
        return x

    @staticmethod
    def encode_move(move):
        y = torch.zeros(64)
        y[chess.parse_square(move[-2:])] = 1
        return y

    def encode(self, games):
        x, y, results = [], [], []
        for game in games:
            board = chess.Board()
            result = game.headers['Result']
            for move in game.mainline_moves():
                board.push_san(move)
                x.append(self.state(board))
                y.append(self.encode_move(move))
                results.append(result)

        return torch.Tensor(x), torch.Tensor(y), torch.Tensor(results)

class Net(nn.Module):
    # ((img_size - kern_size + (2 * padding_size))/stride) + 1
    def __init__(self):
        self. model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(6, 256, kernel_size=3, padding=1, stride=1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)),
            ('relu4', nn.ReLU()),
            ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)),
            ('relu5', nn.ReLU()),
            ('conv6', nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)),
            ('relu6', nn.ReLU()),
            ('conv7', nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)),
            ('relu7', nn.ReLU()),
            ('conv8', nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)),
            ('relu8', nn.ReLU()),
        ]))

    def forward(self, x):
        pass


if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor()])
    chess_dataset = ChessDataset(DATA_DIR, transform)
    data_loader = DataLoader(chess_dataset, batch_size=64, shuffle=True, num_workers=6)

    print(torch.utils.data.get_worker_info())
