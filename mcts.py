import chess
import numpy as np

from state import State

CPUCT = 1

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

    def encode_legal_moves(self, board):
        out = np.zeros(64)
        for move in board.legal_moves:
            board.push(move)
            out[move.to_square] = 1
            board.pop()
        return out

    def expand_node(self, probablity):
        board = chess.Board(self.fen)
        out = self.encode_legal_moves(board)
        probablity = probablity * out

        # for move, prob in (board.legal_moves, probablity):
        #     self.children[move] = Node(self, board.fen(), prob)
        for move in board.legal_moves:
            self.children[move] = Node(self, board.fen(), probablity[move.to_square])

    def node_value(self):
        # sqrt over the whole term or just the numerator?
        self. U = CPUCT * self.P * (np.sqrt(self.parent.N) / (self.N + 1))
        return self.Q + self.U

    def best_node_uct(self):
        return max(self.children.items(), key=lambda x: x[1].node_value())

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


class MCTS():

    # do we need MAXVAL?
    def __init__(self, board: chess.Board, net, num_sims):
        self.board = board
        self.root = Node(None, board.fen(), 1)
        self.net = net
        self.num_sims = num_sims

    def playout(self):

        node = self.root
        board = chess.Board(self.board.fen())
        while True:
            if board.is_game_over():
                break

            move, node = node.best_node_uct()
            board.push(chess.Move.from_uci(str(move)))
            state = State(board).encode_board()
            
            policy, value = self.net(state)
            node.expand_node(policy)
            # TODO: should i pass -value? i don't think so
            node.backpropagte(value)

    # TODO consider turns
    # TODO obv i cannot afford to search till the end of the tree
    # (bro i'm runing this on a cpu), so parallelize the search?
    def search(self):
        for _ in range(self.num_sims):

    def get_leaf_node(self):
        node = self.root
        while node.is_leaf():
            node = node.best_node_uct()[1]
        return node
