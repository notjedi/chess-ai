import chess
import numpy as np

from hashlib import sha256
from random import choice
from collections import defaultdict

"""
MCTS

- Selection - using UCT
- Expansion - all the legal moves
- Playouts - use the NN here
- Backpropagate - update the different variables for all nodes(that we traversed thro)
"""

class Node():
    def __init__(self):
        pass

class MCTS():

    def __init__(self, board: chess.Board, net):
        self.board = board
        self.net = net
        self.visited = set()
        self.UCT = defaultdict(float)
        self.MAXVAL = 10000 # TODO

    def best_move(self):
        return self.search(self.board)

    def hash_board(self, board):
        return sha256(board.__str__().encode('utf-8')).hexdigest()

    def search(self, node):
        # TODO consider turns
        # TODO obv i cannot afford to search till the end of the tree
        # (bro i'm runing this on a cpu), add depth maybe? or parallelize the search
        # TODO: we need not wait till the game is over so we update the stats then and there
        if node.is_game_over():
            return self.MAXVAL

        node = self.get_leaf_node(node)
        node = self.rollout_policy(node)
        state = self.hash_board(node)
        result = self.net(node) #TODO: encode board state here
        # update stats

    def get_leaf_node(self, node):
        while self.is_fully_expanded(node):
            node = self.best_uct(node)
        return node

    def rollout_policy(self, node):
        try:
            return choice(list(node.legal_moves))
        except Exception:
            return node

    # TODO: is there a more efficient way to do this?
    # TODO: hash map table for the state and all legal moves?
    def is_fully_expanded(self, node):
        states = set()
        for move in list(node.legal_moves):
            node.push(move)
            states.add(self.hash_board(node))
            node.pop()
        return len(self.visited.intersection(states)) == node.legal_moves.count()

    def best_uct(self, node):
        pass
