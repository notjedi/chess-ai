import os
import torch
import chess
import numpy as np

from torch import nn
from chess import pgn
from glob import glob
from torch import optim
from tqdm import tqdm, trange
from torchsummary import summary
from torchvision import transforms
from collections import OrderedDict
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

DATA_DIR = '/mnt/Seagate/Code/chess-ai/data'

class FCView(nn.Module):
    def __init__(self):
        super(FCView, self).__init__()

    def forward(self, x):
        shape = x.data.size(0)
        x = x.view(shape, -1)
        return x

class ChessDataset(Dataset):

    def __init__(self, data_dir):
        self.x = None
        self.results = {'0-1': -1, '1/2-1/2': 0, '1-0': 1}
        self.values = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6}
        # self.values = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        #                'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6, None: 0}

        # for file in tqdm(glob(data_dir + '/*.pgn')):
        for file in (t := tqdm(glob(data_dir + '/sample.pgn'))):
            t.set_description('parsing %s' % os.path.basename(file))
            with open(file, encoding='utf-8') as games:
                while True:
                    game = pgn.read_game(games)
                    if game is None:
                        break
                    (x, p, v) = self.encode(game)

                    if self.x is None:
                        self.x = x
                        self.policy = p
                        self.value = v
                    else:
                        self.x = np.concatenate((self.x, x))
                        self.policy = np.concatenate((self.policy, p))
                        self.value = np.concatenate((self.value, v))

        self.x = np.array(self.x)
        self.policy = np.array(self.policy)
        self.value = np.array(self.value)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.policy[idx], self.value[idx]

    def state(self, board):
        x = np.zeros((6, 64), dtype=np.uint8)
        for i in range(64):
            piece = board.piece_at(i)
            if piece != None:
                symbol = piece.symbol().upper()
                x[self.values[symbol.upper()]-1][i] = 1 if symbol.isupper() else -1

        return x.reshape(6, 8, 8)

    @staticmethod
    def encode_move(move):
        # y = np.zeros(64)
        move = move.__str__()
        # y[chess.parse_square(move[-2:])] = 1
        # return y
        return chess.parse_square(move[-2:])

    def encode(self, game):
        x, p, v = [], [], []
        board = chess.Board()
        result = self.results[game.headers['Result']]
        for move in game.mainline_moves():
            board.push(move)
            p.append(self.encode_move(move))
            if board.turn:
                x.append(self.state(board))
                v.append(result)
            else:
                x.append(-self.state(board))
                v.append(-result)
        x = np.array(x, dtype=np.float32)
        p = np.array(p, dtype=np.float32)
        v = np.array(v, dtype=np.float32)
        return [x, p, v]


class Net(nn.Module):
    # ((img_size - kern_size + (2 * padding_size))/stride) + 1
    def __init__(self):
        super(Net, self).__init__()

        self. model = nn.Sequential(OrderedDict([
            # (8 - 3 + 2) / 1 + 1
            ('conv1', nn.Conv2d(6, 256, kernel_size=3, padding=1, stride=1)),
            ('relu1', nn.ReLU()),
            ('batch-norm-1', nn.BatchNorm2d(256)),
            ('conv2', nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)),
            ('relu2', nn.ReLU()),
            ('batch-norm-2', nn.BatchNorm2d(256)),
            ('conv3', nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)),
            ('relu3', nn.ReLU()),
            ('batch-norm-3', nn.BatchNorm2d(256)),
            ('conv4', nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)),
            ('relu4', nn.ReLU()),
            ('batch-norm-4', nn.BatchNorm2d(256)),
            ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)),
            ('relu5', nn.ReLU()),
            ('batch-norm-5', nn.BatchNorm2d(256)),
            ('conv6', nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)),
            ('relu6', nn.ReLU()),
            ('batch-norm-6', nn.BatchNorm2d(256)),
            ('conv7', nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)),
            ('relu7', nn.ReLU()),
            ('batch-norm-7', nn.BatchNorm2d(256)),
            ('conv8', nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)),
            ('relu8', nn.ReLU()),
            ('batch-norm-8', nn.BatchNorm2d(256)),
        ]))

        self.policy = nn.Sequential(OrderedDict([
            # (8 - 1 / 1) + 1
            ('conv1-policy', nn.Conv2d(256, 2, kernel_size=1, padding=0, stride=1)),
            ('relu1-policy', nn.ReLU()),
            ('batch-norm-policy-1', nn.BatchNorm2d(2)),
            ('reshape', FCView()),
            ('fc1-policy', nn.Linear(2 * 8 * 8, 64)),
        ]))
        self.fc1_policy = nn.Linear(128, 64)

        self.value = nn.Sequential(OrderedDict([
            # (8 - 3 + 2 / 1) + 1
            # FIX: should i use a kernel of size 1?
            ('conv1-value', nn.Conv2d(256, 2, kernel_size=3, padding=1, stride=1)),
            ('relu1-value', nn.ReLU()),
            ('batch-norm-value-1', nn.BatchNorm2d(2)),
            ('reshape', FCView()),
            ('fc1-policy', nn.Linear(2 * 8 * 8, 64)),
            ('relu2-value', nn.ReLU()),
            ('fc2-policy', nn.Linear(64, 1)),
        ]))

    def forward(self, x):
        x = self.model(x)
        policy = F.softmax(self.policy(x), dim=1)
        value = F.tanh(self.value(x))
        return policy, value.view(-1)


def plot(loss1, loss2):
    plt.plot(loss1)
    plt.plot(loss2)


if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor()])
    chess_dataset = ChessDataset(DATA_DIR)
    data_loader = DataLoader(chess_dataset, batch_size=64, shuffle=True, num_workers=6, drop_last=True)
    torch.autograd.set_detect_anomaly(True)

    net = Net()
    opt = optim.Adam(net.parameters())
    summary(net, input_size=(6, 8, 8))

    EPOCHS = 10
    for epoch in trange(EPOCHS):
        policy_losses = []
        value_losses = []
        for (data, policy, value) in data_loader:
            opt.zero_grad()
            policy_pred, value_pred = net(data)

            policy_loss = F.nll_loss(torch.log(policy_pred), policy.long())
            value_loss = F.mse_loss(value_pred, value)
            policy_loss.backward(retain_graph=True)
            value_loss.backward()
            opt.step()

            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            # t.set_description('policy loss %.2f value loss %.2f' % (policy_loss, value_loss))
        plot(policy_losses, value_losses)
