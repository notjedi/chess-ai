import torch
import numpy as np

from torch import nn
from tqdm import tqdm, trange
from torch.nn import functional as F
from torch.utils.data import Dataset

from config import N_LABELS, REPORT, device

EPOCHS = 3

# https://web.stanford.edu/~surag/posts/alphazero.html
# https://int8.io/monte-carlo-tree-search-beginners-guide/
# https://www.datahubbs.com/two-headed-a2c-network-in-pytorch/
# https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/
# https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/
# https://github.com/DylanSnyder31/AlphaZero-Chess/blob/660f50f56dc8772cceaf46d33a97aed8b0775fa8/Reinforcement_Learning/Monte_Carlo_Search_Tree/MCTS_main.py

class ChessDataset(Dataset):

    def __init__(self, x, policy, value):
        self.x, self.policy, self.value = np.empty((0, 6, 8, 8), dtype=np.float16), np.empty(0, dtype=np.float16), np.empty(0, dtype=np.float16)

        self.x = torch.from_numpy(np.concatenate((self.x, x), axis=0)).to(device)
        self.policy = torch.from_numpy(np.concatenate((self.policy, policy), axis=0)).to(device)
        self.value = torch.from_numpy(np.concatenate((self.value, value), axis=0)).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.policy[idx], self.value[idx]


class FCView(nn.Module):

    def __init__(self):
        super(FCView, self).__init__()

    def forward(self, x):
        shape = x.data.size(0)
        x = x.view(shape, -1)
        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.conv(x.float()))
        return self.batch_norm(x)


class ResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(F.relu(out))
        out = F.relu(self.conv2(out))
        # skip connection
        # https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/5
        out = out + x
        return self.bn2(out)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class OutBlock(nn.Module):

    def __init__(self):
        super(OutBlock, self).__init__()
        self.reshape = FCView()
        # (8 - 1 / 1) + 1
        self.conv_block_policy = ConvBlock(256, 2, 1, 0, 1)
        self.fc_policy = nn.Linear(2 * 8 * 8, N_LABELS)

        # (8 - 3 + 2 / 1) + 1
        self.conv_block_value = ConvBlock(256, 2, 3, 1, 1)
        self.fc1_value = nn.Linear(2 * 8 * 8, 64)
        self.fc2_value = nn.Linear(64, 1)

    def forward(self, x):
        policy = self.reshape(self.conv_block_policy(x))
        policy = self.fc_policy(policy)
        
        value = self.reshape(self.conv_block_value(x))
        value = F.relu(self.fc1_value(value))
        value = self.fc2_value(value)
        # return F.softmax(policy, dim=1).view(-1), torch.tanh(value).view(-1)
        return F.softmax(policy, dim=1), torch.tanh(value).view(-1)


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, policy_pred, policy, value_pred, value):
        policy_loss = F.nll_loss(torch.log(policy_pred), policy.long())
        value_loss = F.mse_loss(value_pred, value.float())
        return policy_loss, value_loss


class Net(nn.Module):

    # ((img_size - kern_size + (2 * padding_size))/stride) + 1
    def __init__(self):
        super(Net, self).__init__()
        self.conv = ConvBlock(6, 256, 3, 1, 1)
        for block in range(0, 10):
            setattr(self, "res-block-{}".format(block+1), ResBlock(256, 256, 3, 1, 1))
        self.out_block = OutBlock()
        self.initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        for block in range(0, 10):
            x = getattr(self, "res-block-{}".format(block+1))(x)
        policy, value = self.out_block(x)
        return policy, value

    # https://www.youtube.com/watch?v=xWQ-p_o0Uik
    # https://pytorch.org/docs/stable/nn.init.html
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, ResBlock):
                m.initialize_weights()

    def fit(self, data_loader, opt, scheduler, loss, writer, step):

        for _ in trange(EPOCHS):
            for data, policy, value in (t:=tqdm(data_loader)):

                opt.zero_grad(set_to_none=True)
                policy_pred, value_pred = None, None
                policy_pred, value_pred = self.forward(data)

                policy_loss, value_loss = loss(policy_pred, policy, value_pred, value)
                total_loss = 0.9 * policy_loss + 0.1 * value_loss
                # https://github.com/PyTorchLightning/pytorch-lightning/issues/2645
                total_loss.backward()
                opt.step()
                scheduler.step()

                if step % REPORT == 0:
                    writer.add_scalar("Policy Loss/train", policy_loss, step)
                    writer.add_scalar("Value Loss/train", value_loss, step)
                    writer.add_scalar("Learning Rate/train", scheduler.get_last_lr()[0], step)
                    t.set_description('policy loss %.4f value loss %.4f' % (policy_loss, value_loss))
                step += 1

            writer.flush()
        return step
