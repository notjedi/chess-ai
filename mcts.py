import torch
import chess
import numpy as np

from state import State

CPUCT = 1
RESULTS = {'0-1': -1, '1/2-1/2': 0, '1-0': 1}

class Node():
    """
    Class to represent each node in the tree (a move from parent node to self)
    Contains a node's relevant information such as N, P, Q, U
    N = Total number of times the node has been visited
    P = Probablity of selecting this node(move) from the parent node
    Q = Total wins / Total visits ratio (win ratio = Wins / N)
    U = a function that controls the Exploration part of the total_value
    total_value = Exploitation + Exploration (Q + U)
    """
    def __init__(self, parent, fen, prob):
        board = chess.Board(fen)
        self.parent = parent
        self.fen = fen
        self.children = {}
        self.state = State(board).encode_board()
        self.N = 0
        self.Q = 0
        self.U = 0
        self.P = prob

    def expand(self, probablity):
        board = chess.Board(self.fen)
        for move in board.legal_moves:
            self.children[move] = Node(self, self.fen, probablity[move.to_square])

    def value(self):
        # sqrt over the whole term or just the numerator?
        self. U = CPUCT * self.P * (np.sqrt(self.parent.N) / (self.N + 1))
        return self.Q + self.U

    def best_node_uct(self):
        return max(self.children.items(), key=lambda x: x[1].value())

    def is_leaf(self):
        return len(self.children) == 0

    def update_leaf_value(self, value):
        self.N += 1
        self.Q = ((self.Q + value) * 1.0) / self.N

    def backpropagte(self, value):
        # -value because the parent node is played by the opponent
        if self.parent != None:
            self.parent.backpropagte(-value)
        self.update_leaf_value(value)

    def deleteNodes(self):
        for child in self.children.values():
            child.deleteNodes()


class MCTS():

    def __init__(self, board: chess.Board, net, num_sims):
        self.board = board
        self.root = Node(None, board.fen(), 1)
        self.net = net
        self.num_sims = num_sims

    def playout(self):

        # select
        node = self.get_leaf_node()

        # expand
        board = chess.Board(node.fen)
        state = torch.tensor(State(board).encode_board()[np.newaxis])
        # consider turns
        if not board.turn:
            state = -state
        policy, value = self.net(state)
        node.expand(policy.detach().numpy().squeeze())

        # backpropagte
        if board.is_game_over():
            node.backpropagte(RESULTS[board.result()])
        else:
            # TODO: should i pass -value? i don't think so
            # TODO: value should be 0 or 1
            # if int(round(value)) == 1:
            #     value = 1
            # else:
            #     value = 0
            node.backpropagte(value)

    def choose_move(self):
        for _ in range(self.num_sims):
            self.playout()
        selected = self.root.best_node_uct()
        self.root.deleteNodes()
        return selected

    def get_leaf_node(self):
        node = self.root
        while not node.is_leaf():
            node = node.best_node_uct()[1]
        return node
