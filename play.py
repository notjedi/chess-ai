import chess
import torch
import base64

from model import Net
from mcts import MCTS
from chess import svg
from flask import Flask, request
from config import device

DEBUG = 1
app = Flask(__name__)
net = Net()
net.to(device)
net.eval()
board = chess.Board()
html = open('static/index.html', 'r').read()

def encode_svg(board):
  return base64.b64encode(svg.board(board).encode('utf-8')).decode('utf-8')

@app.route('/')
def main():
    net.load_state_dict(torch.load('/mnt/Seagate/Code/chess-ai/model/model.pth'))
    net.eval()
    page = html.replace('repr', encode_svg(board))
    return page

@app.route('/move')
def move():
    if board.is_game_over():
        print("Game Over", board.result())
        return app.response_class(
          response="Game Over",
          status=200
        )

    if board.turn:
        # i have to play
        move, value = request.args.get('move', default=''), None
        if DEBUG:
            print(move, board.fen())
    else:
        # the model has to play
        mcts = MCTS(board, net, 500)
        move, node = mcts.choose_move()
        move = move.uci()
        value = node.value()
        del mcts
        if DEBUG:
            print(move, value, board.fen())

    try:
        board.push_san(move)
    except ValueError:
        print(f'No legal move for {move} in {board.fen()}')

    return app.response_class(
      response=encode_svg(board),
      status=200
    )


if __name__ == '__main__':
    app.run(debug=True)
