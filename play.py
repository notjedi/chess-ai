import chess
import base64
import traceback

from train import Net
from mcts import MCTS
from chess import svg
from flask import Flask, request

DEBUG = 1
app = Flask(__name__)
net = Net()
board = chess.Board()
html = open('static/index.html', 'r').read()

def encode_svg(board):
  return base64.b64encode(svg.board(board).encode('utf-8')).decode('utf-8')

@app.route('/')
def main():
    page = html.replace('repr', encode_svg(board))
    return page

@app.route('/move')
def move():
    if board.is_game_over():
        return app.response_class(
          response="game over",
          status=200
        )

    # i have to play
    move, value = request.args.get('move', default=''), None
    if not board.turn:
        # the net has to play
        mcts = MCTS(board, net, 500)
        move, value = mcts.choose_move()
        move = move.__str__()
        value = value.value()

    try:
        # board.push(chess.Move(chess.parse_square(move[:2]), chess.parse_square(move[2:])))
        board.push_san(move)
    except:
        traceback.print_exc()
    if DEBUG:
        print(move, value)

    return app.response_class(
      response=encode_svg(board),
      status=200
    )


if __name__ == '__main__':
    net.eval()
    # TODO: load the model here
    # net.load_state_dict()
    app.run(debug=True)
